use crate::config::{FftConfig, FftPrecision};
use crate::diagnostics::{
    helper_buffer_requirements_from_graph, stage_summaries_for_route, stage_summaries_from_graph,
    FftBlocker, FftBlockerKind, FftBufferRequirement, FftDeviceLimits, FftDiagnostics,
    FftRouteSummary, FftStageSummary,
};
use crate::error::{FftError, FftExecutionError, FftPlanCreationError, Result};
use crate::runtime::axis_policy::AxisKind;
use crate::runtime::buffer_view::{BufferView, FftIoView};
use crate::runtime::c2c::{C2cPlan, C2cRoute};
use crate::runtime::large_graph::{ElementFormat, LargeExecutionGraph};
use crate::runtime::large_policy::{
    LargeExecutionKind, LargePolicyLimits, LargeRouteMode, LargeRoutingPolicy,
};
use crate::runtime::logical_io::{
    BoundLogicalIo, FftEndpointFormat, FftLogicalLayout, FftLogicalView,
};
use crate::runtime::real::{C2rPlan, R2cPlan};
use crate::runtime::stage_executor::StageExecutor;
use crate::runtime::window_scheduler::{SchedulerLimits, WindowScheduler};
use crate::tuning::FftTuningSummary;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftTransformKind {
    C2c,
    R2c,
    C2r,
}

/// Public plan handle for supported FFT transforms.
pub struct FftPlan {
    inner: FftPlanInner,
}

enum FftPlanInner {
    C2c(C2cPlan),
    R2c(R2cPlan),
    C2r(C2rPlan),
}

struct GpuPlanCreationErrorScopes {
    out_of_memory: wgpu::ErrorScopeGuard,
    internal: wgpu::ErrorScopeGuard,
    validation: wgpu::ErrorScopeGuard,
}

#[derive(Clone, Copy)]
struct PlanResourceAllocationContext {
    route: &'static str,
    resource: &'static str,
    requested_bytes: u64,
    buffer_count: usize,
}

impl GpuPlanCreationErrorScopes {
    fn push(device: &wgpu::Device) -> Self {
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self {
            out_of_memory,
            internal,
            validation,
        }
    }

    async fn pop_error(self) -> Option<(&'static str, String)> {
        // Error scopes are a stack. Pop every scope in strict reverse order,
        // even when plan construction already returned a synchronous error.
        // Initiate every pop before awaiting. Yielding between pops would let
        // another browser operation interleave scopes above the lower guards.
        let validation_pop = self.validation.pop();
        let internal_pop = self.internal.pop();
        let out_of_memory_pop = self.out_of_memory.pop();
        let validation_error = validation_pop.await;
        let internal_error = internal_pop.await;
        let out_of_memory_error = out_of_memory_pop.await;
        if let Some(error) = validation_error {
            Some(("validation", error.to_string()))
        } else if let Some(error) = internal_error {
            Some(("internal", error.to_string()))
        } else if let Some(error) = out_of_memory_error {
            Some(("out-of-memory", error.to_string()))
        } else {
            None
        }
    }
}

impl PlanResourceAllocationContext {
    fn for_config(_config: &FftConfig) -> Self {
        Self {
            route: "c2c-plan",
            resource: "plan-owned-gpu-resources",
            requested_bytes: 0,
            buffer_count: 0,
        }
    }

    fn for_plan(plan: &FftPlan, fallback: Self) -> Self {
        let diagnostics = plan.diagnostics();
        // An error scope covers the complete constructor, so it cannot safely
        // attribute a device error to one particular helper. Report the full
        // plan-owned inventory rather than mislabeling a ring/workspace failure
        // as an arena allocation failure.
        let helper_requirements = diagnostics
            .buffer_requirements()
            .iter()
            .filter(|requirement| {
                requirement.role.starts_with("helper:") || requirement.role == "workspace"
            })
            .collect::<Vec<_>>();
        if helper_requirements.is_empty() {
            return Self {
                route: plan.large_routing_policy().execution_kind().as_str(),
                ..fallback
            };
        }
        Self {
            route: plan.large_routing_policy().execution_kind().as_str(),
            resource: "plan-owned-gpu-resources",
            requested_bytes: helper_requirements.iter().fold(0u64, |total, requirement| {
                total.saturating_add(requirement.required_bytes)
            }),
            buffer_count: helper_requirements.len(),
        }
    }

    fn into_error(self, kind: &'static str, details: String) -> FftError {
        FftError::GpuPlanResourceAllocationFailed {
            route: self.route,
            resource: self.resource,
            requested_bytes: self.requested_bytes,
            buffer_count: self.buffer_count,
            kind,
            details,
        }
    }
}

async fn finish_checked_c2c_plan_creation(
    result: Result<FftPlan>,
    fallback: PlanResourceAllocationContext,
    scopes: GpuPlanCreationErrorScopes,
) -> Result<FftPlan> {
    let captured_error = scopes.pop_error().await;
    if let Some((kind, details)) = captured_error {
        let context = result
            .as_ref()
            .map(|plan| PlanResourceAllocationContext::for_plan(plan, fallback))
            .unwrap_or(fallback);
        Err(context.into_error(kind, details))
    } else {
        result
    }
}

impl FftPlan {
    pub fn c2c(device: &wgpu::Device, queue: &wgpu::Queue, config: FftConfig) -> Result<Self> {
        Ok(Self {
            inner: FftPlanInner::C2c(C2cPlan::new(device, queue, config)?),
        })
    }

    pub fn c2c_with_diagnostics(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
    ) -> std::result::Result<Self, FftPlanCreationError> {
        let tuning = config.tuning().clone();
        Self::c2c(device, queue, config)
            .map_err(|error| FftPlanCreationError::from_error_with_tuning(error, "c2c", tuning))
    }

    /// Creates a C2C plan while converting scoped GPU validation, internal,
    /// and out-of-memory failures into a structured [`FftError`].
    pub async fn c2c_checked(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
    ) -> Result<Self> {
        let fallback = PlanResourceAllocationContext::for_config(&config);
        let scopes = GpuPlanCreationErrorScopes::push(device);
        let result = Self::c2c(device, queue, config);
        finish_checked_c2c_plan_creation(result, fallback, scopes).await
    }

    /// Checked C2C plan creation with structured diagnostics attached to any
    /// synchronous or scoped GPU failure.
    pub async fn c2c_checked_with_diagnostics(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
    ) -> std::result::Result<Self, FftPlanCreationError> {
        let tuning = config.tuning().clone();
        Self::c2c_checked(device, queue, config)
            .await
            .map_err(|error| FftPlanCreationError::from_error_with_tuning(error, "c2c", tuning))
    }

    #[doc(hidden)]
    pub fn c2c_with_large_policy_limits_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        Ok(Self {
            inner: FftPlanInner::C2c(C2cPlan::new_with_large_policy_limits_for_testing(
                device, queue, config, limits,
            )?),
        })
    }

    #[doc(hidden)]
    pub fn c2c_with_large_policy_limits_and_burst_depth_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
        burst_depth: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: FftPlanInner::C2c(
                C2cPlan::new_with_large_policy_limits_and_burst_depth_for_testing(
                    device,
                    queue,
                    config,
                    limits,
                    burst_depth,
                )?,
            ),
        })
    }

    #[doc(hidden)]
    pub async fn c2c_checked_with_large_policy_limits_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let fallback = PlanResourceAllocationContext::for_config(&config);
        let scopes = GpuPlanCreationErrorScopes::push(device);
        let result = Self::c2c_with_large_policy_limits_for_testing(device, queue, config, limits);
        finish_checked_c2c_plan_creation(result, fallback, scopes).await
    }

    #[doc(hidden)]
    pub async fn c2c_checked_with_large_policy_limits_and_burst_depth_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
        burst_depth: usize,
    ) -> Result<Self> {
        let fallback = PlanResourceAllocationContext::for_config(&config);
        let scopes = GpuPlanCreationErrorScopes::push(device);
        let result = Self::c2c_with_large_policy_limits_and_burst_depth_for_testing(
            device,
            queue,
            config,
            limits,
            burst_depth,
        );
        finish_checked_c2c_plan_creation(result, fallback, scopes).await
    }

    pub fn r2c(device: &wgpu::Device, queue: &wgpu::Queue, config: FftConfig) -> Result<Self> {
        Ok(Self {
            inner: FftPlanInner::R2c(R2cPlan::new(device, queue, config)?),
        })
    }

    pub fn r2c_with_diagnostics(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
    ) -> std::result::Result<Self, FftPlanCreationError> {
        let tuning = config.tuning().clone();
        Self::r2c(device, queue, config)
            .map_err(|error| FftPlanCreationError::from_error_with_tuning(error, "r2c", tuning))
    }

    #[doc(hidden)]
    pub fn r2c_with_large_policy_limits_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        Ok(Self {
            inner: FftPlanInner::R2c(R2cPlan::new_with_large_policy_limits_for_testing(
                device, queue, config, limits,
            )?),
        })
    }

    pub fn c2r(device: &wgpu::Device, queue: &wgpu::Queue, config: FftConfig) -> Result<Self> {
        Ok(Self {
            inner: FftPlanInner::C2r(C2rPlan::new(device, queue, config)?),
        })
    }

    pub fn c2r_with_diagnostics(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
    ) -> std::result::Result<Self, FftPlanCreationError> {
        let tuning = config.tuning().clone();
        Self::c2r(device, queue, config)
            .map_err(|error| FftPlanCreationError::from_error_with_tuning(error, "c2r", tuning))
    }

    #[doc(hidden)]
    pub fn c2r_with_large_policy_limits_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        Ok(Self {
            inner: FftPlanInner::C2r(C2rPlan::new_with_large_policy_limits_for_testing(
                device, queue, config, limits,
            )?),
        })
    }

    pub fn kind(&self) -> FftTransformKind {
        match &self.inner {
            FftPlanInner::C2c(_) => FftTransformKind::C2c,
            FftPlanInner::R2c(_) => FftTransformKind::R2c,
            FftPlanInner::C2r(_) => FftTransformKind::C2r,
        }
    }

    pub fn config(&self) -> FftConfig {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.config(),
            FftPlanInner::R2c(plan) => plan.config(),
            FftPlanInner::C2r(plan) => plan.config(),
        }
    }

    pub fn factors(&self) -> &[usize] {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.factors(),
            FftPlanInner::R2c(plan) => plan.factors(),
            FftPlanInner::C2r(plan) => plan.factors(),
        }
    }

    pub fn axis_factors(&self) -> &[Vec<usize>] {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.axis_factors(),
            FftPlanInner::R2c(plan) => plan.axis_factors(),
            FftPlanInner::C2r(plan) => plan.axis_factors(),
        }
    }

    pub fn axis_kinds(&self) -> &[AxisKind] {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.axis_kinds(),
            FftPlanInner::R2c(plan) => plan.axis_kinds(),
            FftPlanInner::C2r(plan) => plan.axis_kinds(),
        }
    }

    pub fn route(&self) -> C2cRoute {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.route(),
            FftPlanInner::R2c(plan) => plan.route(),
            FftPlanInner::C2r(plan) => plan.route(),
        }
    }

    pub fn large_routing_policy(&self) -> &LargeRoutingPolicy {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.large_routing_policy(),
            FftPlanInner::R2c(plan) => plan.large_routing_policy(),
            FftPlanInner::C2r(plan) => plan.large_routing_policy(),
        }
    }

    pub fn workspace_size_bytes(&self) -> u64 {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.workspace_size_bytes(),
            FftPlanInner::R2c(plan) => plan.workspace_size_bytes(),
            FftPlanInner::C2r(plan) => plan.workspace_size_bytes(),
        }
    }

    fn twiddle_lut_storage_bytes(&self) -> u64 {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.twiddle_lut_storage_bytes(),
            FftPlanInner::R2c(plan) => plan.twiddle_lut_storage_bytes(),
            FftPlanInner::C2r(plan) => plan.twiddle_lut_storage_bytes(),
        }
    }

    pub fn required_input_buffer_size_bytes(&self) -> u64 {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.required_buffer_size_bytes(),
            FftPlanInner::R2c(plan) => plan.required_input_buffer_size_bytes(),
            FftPlanInner::C2r(plan) => plan.required_input_buffer_size_bytes(),
        }
    }

    pub fn required_output_buffer_size_bytes(&self) -> u64 {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.required_buffer_size_bytes(),
            FftPlanInner::R2c(plan) => plan.required_output_buffer_size_bytes(),
            FftPlanInner::C2r(plan) => plan.required_output_buffer_size_bytes(),
        }
    }

    pub fn required_buffer_size_bytes(&self) -> u64 {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.required_buffer_size_bytes(),
            FftPlanInner::R2c(plan) => plan.required_buffer_size_bytes(),
            FftPlanInner::C2r(plan) => plan.required_buffer_size_bytes(),
        }
    }

    pub fn packed_shape(&self) -> Option<&[usize]> {
        match &self.inner {
            FftPlanInner::C2c(_) => None,
            FftPlanInner::R2c(plan) => Some(plan.packed_shape()),
            FftPlanInner::C2r(plan) => Some(plan.packed_shape()),
        }
    }

    pub fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        self.execute_views(
            device,
            encoder,
            BufferView::whole(input),
            BufferView::whole(output),
        )
        .expect("whole-buffer FFT execution should satisfy buffer view validation");
    }

    pub fn execute_checked(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> std::result::Result<(), FftExecutionError> {
        self.execute_with_diagnostics(device, encoder, input, output)
    }

    pub fn execute_with_diagnostics(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> std::result::Result<(), FftExecutionError> {
        self.execute_views_with_diagnostics(
            device,
            encoder,
            BufferView::whole(input),
            BufferView::whole(output),
        )
    }

    pub fn execute_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        workspace: &wgpu::Buffer,
    ) -> Result<()> {
        self.execute_views_with_workspace(
            device,
            encoder,
            BufferView::whole(input),
            BufferView::whole(output),
            BufferView::whole(workspace),
        )
    }

    pub fn execute_checked_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        workspace: &wgpu::Buffer,
    ) -> std::result::Result<(), FftExecutionError> {
        self.execute_with_workspace_diagnostics(device, encoder, input, output, workspace)
    }

    pub fn execute_with_workspace_diagnostics(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        workspace: &wgpu::Buffer,
    ) -> std::result::Result<(), FftExecutionError> {
        self.execute_views_with_workspace_diagnostics(
            device,
            encoder,
            BufferView::whole(input),
            BufferView::whole(output),
            BufferView::whole(workspace),
        )
    }

    pub fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        input.validate_min_size(self.required_input_buffer_size_bytes())?;
        output.validate_min_size(self.required_output_buffer_size_bytes())?;
        self.execute_logical_views(
            device,
            encoder,
            FftLogicalView::contiguous(input),
            FftLogicalView::contiguous(output),
        )
    }

    pub fn execute_views_with_diagnostics(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> std::result::Result<(), FftExecutionError> {
        let diagnostic_input = FftLogicalView::contiguous(input.clone());
        let diagnostic_output = FftLogicalView::contiguous(output.clone());
        self.execute_views(device, encoder, input, output)
            .map_err(|error| {
                self.execution_error_for_logical_views(
                    device,
                    &diagnostic_input,
                    &diagnostic_output,
                    error,
                )
            })
    }

    pub fn execute_views_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: BufferView<'_>,
    ) -> Result<()> {
        input.validate_min_size(self.required_input_buffer_size_bytes())?;
        output.validate_min_size(self.required_output_buffer_size_bytes())?;
        self.execute_logical_views_with_workspace(
            device,
            encoder,
            FftLogicalView::contiguous(input),
            FftLogicalView::contiguous(output),
            workspace,
        )
    }

    pub fn execute_views_with_workspace_diagnostics(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: BufferView<'_>,
    ) -> std::result::Result<(), FftExecutionError> {
        let diagnostic_input = FftLogicalView::contiguous(input.clone());
        let diagnostic_output = FftLogicalView::contiguous(output.clone());
        let diagnostic_workspace = workspace.clone();
        self.execute_views_with_workspace(device, encoder, input, output, workspace)
            .map_err(|error| {
                self.execution_error_for_logical_views_with_workspace(
                    device,
                    &diagnostic_input,
                    &diagnostic_output,
                    &diagnostic_workspace,
                    error,
                )
            })
    }

    pub fn execute_io_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
    ) -> Result<()> {
        self.execute_logical_views(
            device,
            encoder,
            FftLogicalView::from_io_view(input),
            FftLogicalView::from_io_view(output),
        )
    }

    pub fn execute_io_views_with_diagnostics(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
    ) -> std::result::Result<(), FftExecutionError> {
        let diagnostic_input = FftLogicalView::from_io_view(input.clone());
        let diagnostic_output = FftLogicalView::from_io_view(output.clone());
        self.execute_io_views(device, encoder, input, output)
            .map_err(|error| {
                self.execution_error_for_logical_views(
                    device,
                    &diagnostic_input,
                    &diagnostic_output,
                    error,
                )
            })
    }

    pub fn execute_io_views_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
        workspace: BufferView<'_>,
    ) -> Result<()> {
        self.execute_logical_views_with_workspace(
            device,
            encoder,
            FftLogicalView::from_io_view(input),
            FftLogicalView::from_io_view(output),
            workspace,
        )
    }

    pub fn execute_io_views_with_workspace_diagnostics(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
        workspace: BufferView<'_>,
    ) -> std::result::Result<(), FftExecutionError> {
        let diagnostic_input = FftLogicalView::from_io_view(input.clone());
        let diagnostic_output = FftLogicalView::from_io_view(output.clone());
        let diagnostic_workspace = workspace.clone();
        self.execute_io_views_with_workspace(device, encoder, input, output, workspace)
            .map_err(|error| {
                self.execution_error_for_logical_views_with_workspace(
                    device,
                    &diagnostic_input,
                    &diagnostic_output,
                    &diagnostic_workspace,
                    error,
                )
            })
    }

    pub fn execute_logical_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftLogicalView<'_>,
        output: FftLogicalView<'_>,
    ) -> Result<()> {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.execute_logical_views(device, encoder, input, output),
            FftPlanInner::R2c(plan) => plan.execute_logical_views(device, encoder, input, output),
            FftPlanInner::C2r(plan) => plan.execute_logical_views(device, encoder, input, output),
        }
    }

    pub fn execute_logical_views_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftLogicalView<'_>,
        output: FftLogicalView<'_>,
        workspace: BufferView<'_>,
    ) -> Result<()> {
        match &self.inner {
            FftPlanInner::C2c(plan) => {
                plan.execute_logical_views_with_workspace(device, encoder, input, output, workspace)
            }
            FftPlanInner::R2c(plan) => plan.execute_logical_views(device, encoder, input, output),
            FftPlanInner::C2r(plan) => plan.execute_logical_views(device, encoder, input, output),
        }
    }

    pub fn execute_logical_views_with_workspace_diagnostics(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftLogicalView<'_>,
        output: FftLogicalView<'_>,
        workspace: BufferView<'_>,
    ) -> std::result::Result<(), FftExecutionError> {
        let diagnostic_input = input.clone();
        let diagnostic_output = output.clone();
        let diagnostic_workspace = workspace.clone();
        self.execute_logical_views_with_workspace(device, encoder, input, output, workspace)
            .map_err(|error| {
                self.execution_error_for_logical_views_with_workspace(
                    device,
                    &diagnostic_input,
                    &diagnostic_output,
                    &diagnostic_workspace,
                    error,
                )
            })
    }

    pub fn execute_logical_views_with_diagnostics(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftLogicalView<'_>,
        output: FftLogicalView<'_>,
    ) -> std::result::Result<(), FftExecutionError> {
        let diagnostic_input = input.clone();
        let diagnostic_output = output.clone();
        self.execute_logical_views(device, encoder, input, output)
            .map_err(|error| {
                self.execution_error_for_logical_views(
                    device,
                    &diagnostic_input,
                    &diagnostic_output,
                    error,
                )
            })
    }

    pub fn diagnostics(&self) -> FftDiagnostics {
        let transform = match self.kind() {
            FftTransformKind::C2c => "c2c",
            FftTransformKind::R2c => "r2c",
            FftTransformKind::C2r => "c2r",
        };
        let route = self.route().as_str().to_owned();
        let policy = self.large_routing_policy();
        let config = self.config();
        let precision = config.precision();
        let mut diagnostics = FftDiagnostics::new(FftRouteSummary::from_large_policy(
            transform,
            route.clone(),
            policy,
        ))
        .with_device_limits(FftDeviceLimits::from_policy(policy))
        .with_active_tuning(active_tuning_summary(
            &config,
            policy.max_bind_bytes,
            policy.max_buffer_size,
        ))
        .with_buffer_requirement(FftBufferRequirement::new(
            "input",
            self.required_input_buffer_size_bytes(),
            input_format_for(self.kind(), precision),
        ))
        .with_buffer_requirement(FftBufferRequirement::new(
            "output",
            self.required_output_buffer_size_bytes(),
            output_format_for(self.kind(), precision),
        ));
        if self.workspace_size_bytes() > 0 {
            diagnostics = diagnostics.with_buffer_requirement(FftBufferRequirement::new(
                "workspace",
                self.workspace_size_bytes(),
                complex_endpoint_format_for(precision).as_str(),
            ));
        }
        let twiddle_lut_storage_bytes = self.twiddle_lut_storage_bytes();
        if twiddle_lut_storage_bytes > 0 {
            diagnostics = diagnostics.with_buffer_requirement(FftBufferRequirement::new(
                "helper:twiddle-luts-total",
                twiddle_lut_storage_bytes,
                complex_endpoint_format_for(precision).as_str(),
            ));
        }
        let stages = match self.execution_graph() {
            Ok(graph) => {
                for requirement in helper_buffer_requirements_from_graph(&graph) {
                    diagnostics = diagnostics.with_buffer_requirement(requirement);
                }
                stage_summaries_from_graph(route.clone(), &graph)
            }
            Err(error) => {
                diagnostics = add_error_diagnostic_blockers(diagnostics, &route, &error);
                stage_summaries_for_route(
                    route,
                    policy.execution_kind(),
                    self.required_buffer_size_bytes(),
                )
            }
        };
        for stage in stages {
            diagnostics = diagnostics.with_stage(stage);
        }
        diagnostics
    }

    pub fn diagnostics_for_device(&self, device: &wgpu::Device) -> FftDiagnostics {
        self.diagnostics_for_limits(FftDeviceLimits::from_device(device))
    }

    pub fn diagnostics_for_limits(&self, limits: FftDeviceLimits) -> FftDiagnostics {
        let mut diagnostics = self.diagnostics().with_device_limits(limits);
        let route = self.route().as_str().to_owned();
        match self.execution_graph() {
            Ok(graph) => {
                let scheduler = WindowScheduler::new(scheduler_limits_from_diagnostics(limits));
                let executor = StageExecutor::new(&scheduler);
                for blocker in executor.graph_blockers(route, &graph) {
                    diagnostics = diagnostics.with_blocker(blocker);
                }
            }
            Err(_) => {}
        }
        diagnostics
    }

    pub fn diagnostics_for_workspace(
        &self,
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        workspace: &wgpu::Buffer,
    ) -> FftDiagnostics {
        self.diagnostics_for_views_with_workspace(
            device,
            BufferView::whole(input),
            BufferView::whole(output),
            BufferView::whole(workspace),
        )
    }

    pub fn diagnostics_for_workspace_with_limits(
        &self,
        limits: FftDeviceLimits,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        workspace: &wgpu::Buffer,
    ) -> FftDiagnostics {
        self.diagnostics_for_views_with_workspace_with_limits(
            limits,
            BufferView::whole(input),
            BufferView::whole(output),
            BufferView::whole(workspace),
        )
    }

    pub fn diagnostics_for_views(
        &self,
        device: &wgpu::Device,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> FftDiagnostics {
        self.diagnostics_for_logical_views(
            device,
            &FftLogicalView::contiguous(input),
            &FftLogicalView::contiguous(output),
        )
    }

    pub fn diagnostics_for_views_with_limits(
        &self,
        limits: FftDeviceLimits,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> FftDiagnostics {
        self.diagnostics_for_logical_views_with_limits(
            limits,
            &FftLogicalView::contiguous(input),
            &FftLogicalView::contiguous(output),
        )
    }

    pub fn diagnostics_for_views_with_workspace(
        &self,
        device: &wgpu::Device,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: BufferView<'_>,
    ) -> FftDiagnostics {
        self.diagnostics_for_views_with_workspace_with_limits(
            FftDeviceLimits::from_device(device),
            input,
            output,
            workspace,
        )
    }

    pub fn diagnostics_for_views_with_workspace_with_limits(
        &self,
        limits: FftDeviceLimits,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: BufferView<'_>,
    ) -> FftDiagnostics {
        self.diagnostics_for_logical_views_with_workspace_with_limits(
            limits,
            &FftLogicalView::contiguous(input),
            &FftLogicalView::contiguous(output),
            workspace,
        )
    }

    pub fn diagnostics_for_io_views(
        &self,
        device: &wgpu::Device,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
    ) -> FftDiagnostics {
        let input = FftLogicalView::from_io_view(input);
        let output = FftLogicalView::from_io_view(output);
        self.diagnostics_for_logical_views(device, &input, &output)
    }

    pub fn diagnostics_for_io_views_with_limits(
        &self,
        limits: FftDeviceLimits,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
    ) -> FftDiagnostics {
        let input = FftLogicalView::from_io_view(input);
        let output = FftLogicalView::from_io_view(output);
        self.diagnostics_for_logical_views_with_limits(limits, &input, &output)
    }

    pub fn diagnostics_for_io_views_with_workspace(
        &self,
        device: &wgpu::Device,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
        workspace: BufferView<'_>,
    ) -> FftDiagnostics {
        let input = FftLogicalView::from_io_view(input);
        let output = FftLogicalView::from_io_view(output);
        self.diagnostics_for_logical_views_with_workspace(device, &input, &output, workspace)
    }

    pub fn diagnostics_for_io_views_with_workspace_with_limits(
        &self,
        limits: FftDeviceLimits,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
        workspace: BufferView<'_>,
    ) -> FftDiagnostics {
        let input = FftLogicalView::from_io_view(input);
        let output = FftLogicalView::from_io_view(output);
        self.diagnostics_for_logical_views_with_workspace_with_limits(
            limits, &input, &output, workspace,
        )
    }

    pub fn diagnostics_for_logical_views(
        &self,
        device: &wgpu::Device,
        input: &FftLogicalView<'_>,
        output: &FftLogicalView<'_>,
    ) -> FftDiagnostics {
        self.diagnostics_for_logical_views_with_limits(
            FftDeviceLimits::from_device(device),
            input,
            output,
        )
    }

    pub fn diagnostics_for_logical_views_with_limits(
        &self,
        limits: FftDeviceLimits,
        input: &FftLogicalView<'_>,
        output: &FftLogicalView<'_>,
    ) -> FftDiagnostics {
        let mut diagnostics = self.diagnostics_for_limits(limits);
        let route = self.route().as_str().to_owned();
        let kind = self.kind();
        let config = self.config();
        let precision = config.precision();
        let batch = config.batch() as u64;
        let scheduler = WindowScheduler::new(scheduler_limits_from_diagnostics(limits));
        let input_elements = self.input_elements_per_batch();
        let output_elements = self.output_elements_per_batch();
        let input_bound = input_elements.and_then(|elements| {
            scheduler.bind_logical_io(
                input.clone(),
                input_endpoint_format_for(kind, precision),
                elements,
                batch,
            )
        });
        let output_bound = output_elements.and_then(|elements| {
            scheduler.bind_logical_io(
                output.clone(),
                output_endpoint_format_for(kind, precision),
                elements,
                batch,
            )
        });

        let execution_kind = self.large_routing_policy().execution_kind();
        let segmented_volume = execution_kind == LargeExecutionKind::SegmentedFullVolume;
        let windowed_volume = matches!(&self.inner, FftPlanInner::C2c(_))
            && matches!(
                execution_kind,
                LargeExecutionKind::OutOfCoreFourStep | LargeExecutionKind::SegmentedFullVolume
            );
        match input_bound {
            Ok(input) => {
                diagnostics = if windowed_volume {
                    add_four_step_bound_logical_io_diagnostics(
                        diagnostics,
                        "input",
                        &route,
                        &input,
                        limits,
                        segmented_volume,
                    )
                } else {
                    add_bound_logical_io_diagnostics(diagnostics, "input", &route, &input, limits)
                };
            }
            Err(error) => {
                diagnostics = add_logical_io_bind_error_blockers(
                    diagnostics,
                    "input",
                    &route,
                    &error,
                    input,
                    self.required_input_buffer_size_bytes(),
                    input_endpoint_format_for(kind, precision),
                    self.input_elements_per_batch().ok(),
                    batch,
                );
            }
        }
        match output_bound {
            Ok(output) => {
                diagnostics = if windowed_volume {
                    add_four_step_bound_logical_io_diagnostics(
                        diagnostics,
                        "output",
                        &route,
                        &output,
                        limits,
                        segmented_volume,
                    )
                } else {
                    add_bound_logical_io_diagnostics(diagnostics, "output", &route, &output, limits)
                };
            }
            Err(error) => {
                diagnostics = add_logical_io_bind_error_blockers(
                    diagnostics,
                    "output",
                    &route,
                    &error,
                    output,
                    self.required_output_buffer_size_bytes(),
                    output_endpoint_format_for(kind, precision),
                    self.output_elements_per_batch().ok(),
                    batch,
                );
            }
        }
        diagnostics
    }

    pub fn diagnostics_for_logical_views_with_workspace(
        &self,
        device: &wgpu::Device,
        input: &FftLogicalView<'_>,
        output: &FftLogicalView<'_>,
        workspace: BufferView<'_>,
    ) -> FftDiagnostics {
        self.diagnostics_for_logical_views_with_workspace_with_limits(
            FftDeviceLimits::from_device(device),
            input,
            output,
            workspace,
        )
    }

    pub fn diagnostics_for_logical_views_with_workspace_with_limits(
        &self,
        limits: FftDeviceLimits,
        input: &FftLogicalView<'_>,
        output: &FftLogicalView<'_>,
        workspace: BufferView<'_>,
    ) -> FftDiagnostics {
        let mut diagnostics = self.diagnostics_for_logical_views_with_limits(limits, input, output);
        let route = self.route().as_str().to_owned();
        for blocker in self.workspace_blockers_for_view(limits, &workspace) {
            diagnostics = diagnostics.with_blocker(blocker.with_default_route(&route));
        }
        diagnostics
    }

    pub(crate) fn execution_graph(&self) -> Result<LargeExecutionGraph> {
        match &self.inner {
            FftPlanInner::C2c(plan) => plan.execution_graph(),
            FftPlanInner::R2c(plan) => plan.execution_graph(),
            FftPlanInner::C2r(plan) => plan.execution_graph(),
        }
    }

    fn input_elements_per_batch(&self) -> Result<u64> {
        let config = self.config();
        elements_per_batch(
            self.required_input_buffer_size_bytes(),
            config.batch() as u64,
            input_endpoint_format_for(self.kind(), config.precision()),
        )
    }

    fn output_elements_per_batch(&self) -> Result<u64> {
        let config = self.config();
        elements_per_batch(
            self.required_output_buffer_size_bytes(),
            config.batch() as u64,
            output_endpoint_format_for(self.kind(), config.precision()),
        )
    }

    fn execution_error_for_logical_views(
        &self,
        device: &wgpu::Device,
        input: &FftLogicalView<'_>,
        output: &FftLogicalView<'_>,
        error: FftError,
    ) -> FftExecutionError {
        self.execution_error_for_logical_views_with_blockers(
            device,
            input,
            output,
            error,
            std::iter::empty(),
        )
    }

    fn execution_error_for_logical_views_with_blockers<I>(
        &self,
        device: &wgpu::Device,
        input: &FftLogicalView<'_>,
        output: &FftLogicalView<'_>,
        error: FftError,
        extra_blockers: I,
    ) -> FftExecutionError
    where
        I: IntoIterator<Item = FftBlocker>,
    {
        let mut diagnostics = self.diagnostics_for_logical_views(device, input, output);
        let route = self.route().as_str().to_owned();
        for blocker in error.diagnostics().blockers() {
            diagnostics =
                with_blocker_if_missing(diagnostics, blocker.clone().with_default_route(&route));
        }
        for blocker in self.io_size_blockers_for_error(&error, input, output) {
            diagnostics = with_blocker_if_missing(diagnostics, blocker.with_default_route(&route));
        }
        for blocker in extra_blockers {
            diagnostics = with_blocker_if_missing(diagnostics, blocker.with_default_route(&route));
        }
        FftExecutionError::new(error, diagnostics)
    }

    fn execution_error_for_logical_views_with_workspace(
        &self,
        device: &wgpu::Device,
        input: &FftLogicalView<'_>,
        output: &FftLogicalView<'_>,
        workspace: &BufferView<'_>,
        error: FftError,
    ) -> FftExecutionError {
        let mut diagnostics = self.diagnostics_for_logical_views_with_workspace(
            device,
            input,
            output,
            workspace.clone(),
        );
        let route = self.route().as_str().to_owned();
        for blocker in error.diagnostics().blockers() {
            diagnostics =
                with_blocker_if_missing(diagnostics, blocker.clone().with_default_route(&route));
        }
        for blocker in self.io_size_blockers_for_error(&error, input, output) {
            diagnostics = with_blocker_if_missing(diagnostics, blocker.with_default_route(&route));
        }
        if let Some(blocker) = self.workspace_blocker_for_error(&error, workspace) {
            diagnostics = with_blocker_if_missing(diagnostics, blocker.with_default_route(&route));
        }
        FftExecutionError::new(error, diagnostics)
    }

    fn io_size_blockers_for_error(
        &self,
        error: &FftError,
        input: &FftLogicalView<'_>,
        output: &FftLogicalView<'_>,
    ) -> Vec<FftBlocker> {
        let config = self.config();
        let precision = config.precision();
        let batch = config.batch() as u64;
        let mut blockers = logical_view_size_blockers(
            "input",
            error,
            input,
            self.required_input_buffer_size_bytes(),
            input_endpoint_format_for(self.kind(), precision),
            self.input_elements_per_batch().ok(),
            batch,
        );
        blockers.extend(logical_view_size_blockers(
            "output",
            error,
            output,
            self.required_output_buffer_size_bytes(),
            output_endpoint_format_for(self.kind(), precision),
            self.output_elements_per_batch().ok(),
            batch,
        ));
        blockers
    }

    fn workspace_blocker_for_error(
        &self,
        error: &FftError,
        workspace: &BufferView<'_>,
    ) -> Option<FftBlocker> {
        match error {
            FftError::WorkspaceTooSmall { required, actual } => Some(
                FftBlocker::new(FftBlockerKind::Workspace, error.to_string())
                    .with_stage("workspace")
                    .with_helper_buffer("workspace")
                    .with_required_bytes(*required)
                    .with_actual_bytes(*actual),
            ),
            FftError::SegmentedWorkspaceUnsupported => Some(
                FftBlocker::new(FftBlockerKind::Workspace, error.to_string())
                    .with_stage("workspace")
                    .with_helper_buffer("workspace")
                    .with_layout("segmented"),
            ),
            FftError::BufferViewMissingUsage { .. } => Some(
                FftBlocker::new(FftBlockerKind::BufferUsage, error.to_string())
                    .with_stage("workspace")
                    .with_helper_buffer("workspace")
                    .with_layout("storage binding")
                    .with_actual_bytes(workspace.size())
                    .with_required_bytes(self.workspace_size_bytes()),
            ),
            FftError::BufferViewOffsetUnaligned { offset, alignment } => Some(
                FftBlocker::new(FftBlockerKind::Alignment, error.to_string())
                    .with_stage("workspace")
                    .with_helper_buffer("workspace")
                    .with_layout("storage binding")
                    .with_required_bytes(*alignment)
                    .with_actual_bytes(*offset),
            ),
            FftError::WindowScheduleUnsupported {
                requested_bytes,
                max_bind_bytes,
                ..
            } => Some(
                FftBlocker::new(FftBlockerKind::DeviceLimit, error.to_string())
                    .with_stage("workspace")
                    .with_helper_buffer("workspace")
                    .with_required_bytes(*requested_bytes)
                    .with_limit_bytes(*max_bind_bytes),
            ),
            FftError::LargeRouteWorkspaceUnsupported { route_mode } => Some(
                FftBlocker::new(FftBlockerKind::Workspace, error.to_string())
                    .with_route(*route_mode)
                    .with_stage("workspace")
                    .with_helper_buffer("workspace"),
            ),
            _ => None,
        }
    }

    fn workspace_blocker_or_fallback(
        &self,
        error: &FftError,
        workspace: &BufferView<'_>,
    ) -> FftBlocker {
        self.workspace_blocker_for_error(error, workspace)
            .unwrap_or_else(|| {
                fallback_workspace_blocker(error, workspace.size(), self.workspace_size_bytes())
            })
    }

    fn workspace_blockers_for_view(
        &self,
        limits: FftDeviceLimits,
        workspace: &BufferView<'_>,
    ) -> Vec<FftBlocker> {
        if let Some(route_mode) = self.caller_workspace_unsupported_route_mode() {
            return vec![self.workspace_blocker_or_fallback(
                &FftError::LargeRouteWorkspaceUnsupported { route_mode },
                workspace,
            )];
        }

        let required = self.workspace_size_bytes();
        if required == 0 {
            return Vec::new();
        }

        if !workspace.is_single_segment() {
            return vec![self.workspace_blocker_or_fallback(
                &FftError::SegmentedWorkspaceUnsupported,
                workspace,
            )];
        }

        if workspace.size() < required {
            return vec![self.workspace_blocker_or_fallback(
                &FftError::WorkspaceTooSmall {
                    required,
                    actual: workspace.size(),
                },
                workspace,
            )];
        }

        let mut blockers = Vec::new();
        if required > limits.max_buffer_size {
            blockers.push(
                FftBlocker::new(
                    FftBlockerKind::HelperBuffer,
                    "workspace requirement exceeds active max buffer size",
                )
                .with_stage("workspace")
                .with_helper_buffer("workspace")
                .with_required_bytes(required)
                .with_limit_bytes(limits.max_buffer_size),
            );
        }

        let Ok(workspace_prefix) = workspace.clone().prefix(required) else {
            return blockers;
        };
        let scheduler = WindowScheduler::new(scheduler_limits_from_diagnostics(limits));
        let workspace_format = complex_element_format_for(self.config().precision());
        if let Err(error) = scheduler.storage_binding_resource(&workspace_prefix, workspace_format)
        {
            if let Some(blocker) = self.workspace_blocker_for_error(&error, workspace) {
                blockers.push(blocker);
            }
        }
        blockers
    }

    fn caller_workspace_unsupported_route_mode(&self) -> Option<&'static str> {
        match &self.inner {
            FftPlanInner::C2c(_) => {
                let route_mode = self.large_routing_policy().route_mode();
                if route_mode == LargeRouteMode::Normal {
                    None
                } else {
                    Some(route_mode.as_str())
                }
            }
            FftPlanInner::R2c(_) | FftPlanInner::C2r(_) => None,
        }
    }
}

fn scheduler_limits_from_diagnostics(limits: FftDeviceLimits) -> SchedulerLimits {
    SchedulerLimits {
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_buffer_size: limits.max_buffer_size,
        storage_alignment: limits.min_storage_buffer_offset_alignment.max(1),
        copy_alignment: 4,
    }
}

fn active_tuning_summary(
    config: &FftConfig,
    effective_max_storage_buffer_binding_size: u64,
    effective_max_buffer_size: u64,
) -> FftTuningSummary {
    let requested = config.tuning().clone();
    let effective = requested
        .clone()
        .with_max_storage_buffer_binding_size(Some(effective_max_storage_buffer_binding_size))
        .with_max_buffer_size(Some(effective_max_buffer_size));
    FftTuningSummary::new(requested, effective)
}

fn fallback_workspace_blocker(error: &FftError, actual: u64, required: u64) -> FftBlocker {
    FftBlocker::new(FftBlockerKind::Workspace, error.to_string())
        .with_stage("workspace")
        .with_helper_buffer("workspace")
        .with_actual_bytes(actual)
        .with_required_bytes(required)
}

fn elements_per_batch(total_bytes: u64, batch: u64, format: FftEndpointFormat) -> Result<u64> {
    if batch == 0 {
        return Err(FftError::ZeroBatch);
    }
    let element_bytes = format.bytes_per_element();
    Ok(total_bytes / batch / element_bytes)
}

fn complex_endpoint_format_for(precision: FftPrecision) -> FftEndpointFormat {
    match precision {
        FftPrecision::F32 => FftEndpointFormat::ComplexF32,
        FftPrecision::F64 => FftEndpointFormat::ComplexF64,
        FftPrecision::Df64 => FftEndpointFormat::ComplexDf64,
    }
}

fn complex_element_format_for(precision: FftPrecision) -> ElementFormat {
    match precision {
        FftPrecision::F32 => ElementFormat::ComplexF32,
        FftPrecision::F64 => ElementFormat::ComplexF64,
        FftPrecision::Df64 => ElementFormat::ComplexDf64,
    }
}

fn input_endpoint_format_for(kind: FftTransformKind, precision: FftPrecision) -> FftEndpointFormat {
    match kind {
        FftTransformKind::C2c => complex_endpoint_format_for(precision),
        FftTransformKind::R2c => FftEndpointFormat::RealF32,
        FftTransformKind::C2r => FftEndpointFormat::PackedComplexF32,
    }
}

fn output_endpoint_format_for(
    kind: FftTransformKind,
    precision: FftPrecision,
) -> FftEndpointFormat {
    match kind {
        FftTransformKind::C2c => complex_endpoint_format_for(precision),
        FftTransformKind::R2c => FftEndpointFormat::PackedComplexF32,
        FftTransformKind::C2r => FftEndpointFormat::RealF32,
    }
}

fn add_logical_io_bind_error_blockers(
    mut diagnostics: FftDiagnostics,
    role: &'static str,
    route: &str,
    error: &FftError,
    view: &FftLogicalView<'_>,
    required_bytes: u64,
    format: FftEndpointFormat,
    elements_per_batch: Option<u64>,
    batch: u64,
) -> FftDiagnostics {
    diagnostics = add_error_diagnostic_blockers(diagnostics, route, error);
    for blocker in logical_view_size_blockers(
        role,
        error,
        view,
        required_bytes,
        format,
        elements_per_batch,
        batch,
    ) {
        diagnostics = diagnostics.with_blocker(blocker.with_default_route(route));
    }
    diagnostics
}

fn add_error_diagnostic_blockers(
    mut diagnostics: FftDiagnostics,
    route: &str,
    error: &FftError,
) -> FftDiagnostics {
    let error_diagnostics = error.diagnostics();
    for blocker in error_diagnostics.blockers() {
        diagnostics = diagnostics.with_blocker(blocker.clone().with_default_route(route));
    }
    diagnostics
}

fn logical_view_size_blockers(
    role: &'static str,
    error: &FftError,
    view: &FftLogicalView<'_>,
    required_bytes_for_role: u64,
    format: FftEndpointFormat,
    elements_per_batch: Option<u64>,
    batch: u64,
) -> Vec<FftBlocker> {
    match error {
        FftError::BufferViewTooSmall { required, actual } => {
            if required_bytes_for_role == *required && view.view().size() == *actual {
                vec![
                    FftBlocker::new(FftBlockerKind::Validation, error.to_string())
                        .with_stage(format!("{role}-logical-view"))
                        .with_layout(logical_view_layout_label(view, elements_per_batch, batch))
                        .with_required_bytes(*required)
                        .with_actual_bytes(*actual),
                ]
            } else {
                Vec::new()
            }
        }
        FftError::BufferLayoutOutOfBounds {
            required_bytes,
            actual_bytes,
        } => {
            let Some(elements_per_batch) = elements_per_batch else {
                return Vec::new();
            };
            let Ok(expected) =
                required_logical_view_bytes(view.layout(), format, elements_per_batch, batch)
            else {
                return Vec::new();
            };
            if expected == *required_bytes && view.view().size() == *actual_bytes {
                vec![FftBlocker::new(FftBlockerKind::Layout, error.to_string())
                    .with_stage(format!("{role}-logical-view"))
                    .with_layout(logical_view_layout_label(
                        view,
                        Some(elements_per_batch),
                        batch,
                    ))
                    .with_required_bytes(*required_bytes)
                    .with_actual_bytes(*actual_bytes)]
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

fn required_logical_view_bytes(
    layout: FftLogicalLayout,
    format: FftEndpointFormat,
    elements_per_batch: u64,
    batch: u64,
) -> Result<u64> {
    layout
        .required_element_span(elements_per_batch, batch)?
        .checked_mul(format.bytes_per_element())
        .ok_or(FftError::BufferLayoutTooLarge {
            value: u64::MAX,
            limit: u64::MAX - 1,
        })
}

fn logical_view_layout_label(
    view: &FftLogicalView<'_>,
    elements_per_batch: Option<u64>,
    batch: u64,
) -> &'static str {
    let segmented = !view.view().is_single_segment();
    let contiguous = elements_per_batch
        .and_then(|elements| view.layout().is_contiguous_for(elements, batch).ok())
        .unwrap_or(false);
    match (contiguous, segmented) {
        (true, false) => "contiguous",
        (true, true) => "segmented",
        (false, false) => "strided",
        (false, true) => "segmented-strided",
    }
}

fn add_bound_logical_io_diagnostics(
    mut diagnostics: FftDiagnostics,
    role: &'static str,
    route: &str,
    io: &BoundLogicalIo<'_>,
    limits: FftDeviceLimits,
) -> FftDiagnostics {
    diagnostics = add_logical_io_stages(diagnostics, role, route, io);
    diagnostics = add_logical_io_alignment_blockers(diagnostics, role, route, io, limits);
    diagnostics = add_logical_io_usage_blockers(diagnostics, role, route, io);
    add_logical_io_limit_blockers(diagnostics, role, route, io, limits)
}

fn add_four_step_bound_logical_io_diagnostics(
    mut diagnostics: FftDiagnostics,
    role: &'static str,
    route: &str,
    io: &BoundLogicalIo<'_>,
    limits: FftDeviceLimits,
    segmented_volume: bool,
) -> FftDiagnostics {
    diagnostics = add_logical_io_stages(diagnostics, role, route, io);
    if segmented_volume && (!io.is_contiguous() || !io.covers_whole_buffers) {
        return diagnostics.with_blocker(
            FftBlocker::new(
                FftBlockerKind::Unsupported,
                "segmented full-volume execution requires zero-offset whole-buffer endpoints",
            )
            .with_route("large-out-of-core")
            .with_stage(format!("{role}-segmented-volume-endpoint"))
            .with_layout(io.layout_kind()),
        );
    }
    if !io.is_contiguous() {
        return diagnostics.with_blocker(
            FftBlocker::new(
                FftBlockerKind::Unsupported,
                "four-step execution does not yet support strided logical I/O",
            )
            .with_route("large-out-of-core")
            .with_stage(if role == "input" {
                "input-strided-pack"
            } else {
                "output-strided-unpack"
            })
            .with_layout(io.layout_kind()),
        );
    }

    let required_usage = if role == "input" {
        wgpu::BufferUsages::COPY_SRC
    } else if segmented_volume {
        wgpu::BufferUsages::COPY_DST
    } else {
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
    };
    let usage_label = if role == "input" {
        "COPY_SRC"
    } else if segmented_volume {
        "COPY_DST"
    } else {
        "COPY_SRC|COPY_DST"
    };
    for range in &io.physical_ranges {
        if range.offset_bytes % 4 != 0 || range.size_bytes % 4 != 0 {
            diagnostics = diagnostics.with_blocker(
                FftBlocker::new(
                    FftBlockerKind::Alignment,
                    format!(
                        "{role} four-step copy window is not 4-byte aligned at offset {} with size {}",
                        range.offset_bytes, range.size_bytes
                    ),
                )
                .with_route("large-out-of-core")
                .with_stage(format!("{role}-four-step-copy-window"))
                .with_layout(io.layout_kind())
                .with_required_bytes(4)
                .with_actual_bytes(range.offset_bytes.saturating_add(range.size_bytes)),
            );
        }
        if !range.buffer.usage().contains(required_usage) {
            diagnostics = diagnostics.with_blocker(
                FftBlocker::new(
                    FftBlockerKind::BufferUsage,
                    logical_io_usage_reason(
                        role,
                        usage_label,
                        range.offset_bytes,
                        range.size_bytes,
                    ),
                )
                .with_route("large-out-of-core")
                .with_stage(format!("{role}-four-step-copy-window"))
                .with_layout(io.layout_kind())
                .with_actual_bytes(range.size_bytes),
            );
        }
        if range.size_bytes > limits.max_buffer_size {
            diagnostics = diagnostics.with_blocker(
                FftBlocker::new(
                    FftBlockerKind::DeviceLimit,
                    format!("{role} four-step physical range exceeds maxBufferSize"),
                )
                .with_route("large-out-of-core")
                .with_stage(format!("{role}-four-step-copy-window"))
                .with_layout(io.layout_kind())
                .with_required_bytes(range.size_bytes)
                .with_limit_bytes(limits.max_buffer_size),
            );
        }
    }
    diagnostics
}

fn add_logical_io_stages(
    mut diagnostics: FftDiagnostics,
    role: &'static str,
    route: &str,
    io: &BoundLogicalIo<'_>,
) -> FftDiagnostics {
    diagnostics = diagnostics.with_stage(FftStageSummary::new(
        format!("{role}-logical-view"),
        format!(
            "logical-io:{}:{}:ranges-{}",
            io.format.as_str(),
            io.layout_kind(),
            io.physical_ranges.len()
        ),
        route.to_owned(),
        Some(io.logical_bytes),
    ));
    if io.is_segmented() {
        diagnostics = diagnostics.with_stage(FftStageSummary::new(
            format!("{role}-segmented-copy-window"),
            "copy-window",
            route.to_owned(),
            Some(io.physical_span_bytes),
        ));
    }
    if !io.is_contiguous() {
        let stage = if role == "input" {
            "strided-pack"
        } else {
            "strided-unpack"
        };
        diagnostics = diagnostics.with_stage(FftStageSummary::new(
            format!("{role}-{stage}"),
            stage,
            route.to_owned(),
            Some(io.logical_bytes.max(io.physical_span_bytes)),
        ));
    }
    diagnostics
}

fn add_logical_io_alignment_blockers(
    mut diagnostics: FftDiagnostics,
    role: &'static str,
    route: &str,
    io: &BoundLogicalIo<'_>,
    limits: FftDeviceLimits,
) -> FftDiagnostics {
    if io.is_segmented() && !io.copy_aligned {
        let offending_range = io
            .physical_ranges
            .iter()
            .find(|range| range.offset_bytes % 4 != 0 || range.size_bytes % 4 != 0)
            .copied();
        let actual_offset = offending_range.map(|range| range.offset_bytes).unwrap_or(0);
        let actual_size = offending_range.map(|range| range.size_bytes).unwrap_or(0);
        diagnostics = diagnostics.with_blocker(
            FftBlocker::new(
                FftBlockerKind::Alignment,
                format!(
                    "{role} logical view has copy-unaligned physical segment at offset {actual_offset} with size {actual_size}"
                ),
            )
            .with_route(route.to_owned())
            .with_stage(format!("{role}-segmented-copy-window"))
            .with_layout(io.layout_kind())
            .with_required_bytes(4)
            .with_actual_bytes(actual_offset.saturating_add(actual_size)),
        );
    }
    if !io.is_segmented() && !io.storage_aligned {
        let actual_offset = io
            .physical_ranges
            .first()
            .map(|range| range.offset_bytes)
            .unwrap_or(0);
        diagnostics = diagnostics.with_blocker(
            FftBlocker::new(
                FftBlockerKind::Alignment,
                format!("{role} logical view is not storage-binding aligned"),
            )
            .with_route(route.to_owned())
            .with_stage(format!("{role}-storage-window"))
            .with_layout(io.layout_kind())
            .with_required_bytes(limits.min_storage_buffer_offset_alignment.max(1))
            .with_actual_bytes(actual_offset),
        );
    }
    diagnostics
}

fn add_logical_io_usage_blockers(
    mut diagnostics: FftDiagnostics,
    role: &'static str,
    route: &str,
    io: &BoundLogicalIo<'_>,
) -> FftDiagnostics {
    let (usage, usage_label, stage) = if io.is_segmented() {
        let usage = if role == "input" {
            wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::COPY_DST
        };
        (
            usage,
            if role == "input" {
                "COPY_SRC"
            } else {
                "COPY_DST"
            },
            format!("{role}-segmented-copy-window"),
        )
    } else if !io.is_contiguous() {
        (
            wgpu::BufferUsages::STORAGE,
            "STORAGE",
            logical_strided_stage(role).to_owned(),
        )
    } else {
        (
            wgpu::BufferUsages::STORAGE,
            "STORAGE",
            format!("{role}-storage-window"),
        )
    };

    for range in &io.physical_ranges {
        if !range.buffer.usage().contains(usage) {
            diagnostics = diagnostics.with_blocker(
                FftBlocker::new(
                    FftBlockerKind::BufferUsage,
                    logical_io_usage_reason(
                        role,
                        usage_label,
                        range.offset_bytes,
                        range.size_bytes,
                    ),
                )
                .with_route(route.to_owned())
                .with_stage(stage.clone())
                .with_layout(io.layout_kind())
                .with_actual_bytes(range.size_bytes),
            );
        }
    }
    diagnostics
}

fn logical_io_usage_reason(
    role: &'static str,
    usage_label: &'static str,
    offset_bytes: u64,
    size_bytes: u64,
) -> String {
    format!(
        "{role} logical view segment at physical offset {offset_bytes} with size {size_bytes} is missing required {usage_label} usage"
    )
}

fn add_logical_io_limit_blockers(
    mut diagnostics: FftDiagnostics,
    role: &'static str,
    route: &str,
    io: &BoundLogicalIo<'_>,
    limits: FftDeviceLimits,
) -> FftDiagnostics {
    let copy_range_bytes = io
        .physical_ranges
        .iter()
        .map(|range| range.size_bytes)
        .collect::<Vec<_>>();
    for blocker in logical_io_limit_blockers(
        LogicalIoLimitFacts {
            role,
            route,
            layout: io.layout_kind(),
            is_contiguous: io.is_contiguous(),
            is_segmented: io.is_segmented(),
            logical_bytes: io.logical_bytes,
            physical_span_bytes: io.physical_span_bytes,
            direct_binding_bytes: direct_storage_binding_bytes(io, limits),
            copy_range_bytes: &copy_range_bytes,
        },
        limits,
    ) {
        diagnostics = diagnostics.with_blocker(blocker);
    }
    diagnostics
}

#[derive(Debug, Clone, Copy)]
struct LogicalIoLimitFacts<'a> {
    role: &'static str,
    route: &'a str,
    layout: &'static str,
    is_contiguous: bool,
    is_segmented: bool,
    logical_bytes: u64,
    physical_span_bytes: u64,
    direct_binding_bytes: Option<u64>,
    copy_range_bytes: &'a [u64],
}

fn logical_io_limit_blockers(
    facts: LogicalIoLimitFacts<'_>,
    limits: FftDeviceLimits,
) -> Vec<FftBlocker> {
    let mut blockers = Vec::new();
    if !facts.is_contiguous {
        push_max_buffer_blocker(
            &mut blockers,
            facts,
            logical_strided_stage(facts.role),
            Some(format!("{}-logical-stage", facts.role)),
            facts.logical_bytes,
            "logical strided staging buffer exceeds active max buffer size",
            limits.max_buffer_size,
        );
        push_storage_binding_blocker(
            &mut blockers,
            facts,
            logical_strided_stage(facts.role),
            Some(format!("{}-logical-stage", facts.role)),
            facts.logical_bytes,
            "logical strided staging buffer exceeds active storage binding limit",
            limits.max_storage_buffer_binding_size,
        );
        if facts.is_segmented {
            push_max_buffer_blocker(
                &mut blockers,
                facts,
                &format!("{}-segmented-strided-physical-stage", facts.role),
                Some(format!("{}-physical-stage", facts.role)),
                facts.physical_span_bytes,
                "segmented strided physical staging buffer exceeds active max buffer size",
                limits.max_buffer_size,
            );
            push_storage_binding_blocker(
                &mut blockers,
                facts,
                logical_strided_stage(facts.role),
                Some(format!("{}-physical-stage", facts.role)),
                facts.physical_span_bytes,
                "segmented strided physical staging buffer exceeds active storage binding limit",
                limits.max_storage_buffer_binding_size,
            );
        }
    }

    if facts.is_segmented {
        for &range_bytes in facts.copy_range_bytes {
            if range_bytes > limits.max_buffer_size {
                blockers.push(
                    FftBlocker::new(
                        FftBlockerKind::DeviceLimit,
                        "segmented copy range exceeds active max buffer size",
                    )
                    .with_route(facts.route.to_owned())
                    .with_stage(format!("{}-segmented-copy-window", facts.role))
                    .with_layout(facts.layout)
                    .with_required_bytes(range_bytes)
                    .with_limit_bytes(limits.max_buffer_size),
                );
            }
        }
    } else if let Some(binding_bytes) = facts.direct_binding_bytes {
        push_storage_binding_blocker(
            &mut blockers,
            facts,
            direct_storage_stage(facts.role, facts.is_contiguous),
            None,
            binding_bytes,
            "logical view storage binding window exceeds active device limit",
            limits.max_storage_buffer_binding_size,
        );
    }

    blockers
}

fn push_max_buffer_blocker(
    blockers: &mut Vec<FftBlocker>,
    facts: LogicalIoLimitFacts<'_>,
    stage: &str,
    helper_buffer: Option<String>,
    required_bytes: u64,
    reason: &'static str,
    max_buffer_size: u64,
) {
    if required_bytes <= max_buffer_size {
        return;
    }
    let mut blocker = FftBlocker::new(FftBlockerKind::HelperBuffer, reason)
        .with_route(facts.route.to_owned())
        .with_stage(stage.to_owned())
        .with_layout(facts.layout)
        .with_required_bytes(required_bytes)
        .with_limit_bytes(max_buffer_size);
    if let Some(helper_buffer) = helper_buffer {
        blocker = blocker.with_helper_buffer(helper_buffer);
    }
    blockers.push(blocker);
}

fn push_storage_binding_blocker(
    blockers: &mut Vec<FftBlocker>,
    facts: LogicalIoLimitFacts<'_>,
    stage: &str,
    helper_buffer: Option<String>,
    required_bytes: u64,
    reason: &'static str,
    max_binding_bytes: u64,
) {
    if required_bytes <= max_binding_bytes {
        return;
    }
    let mut blocker = FftBlocker::new(FftBlockerKind::DeviceLimit, reason)
        .with_route(facts.route.to_owned())
        .with_stage(stage.to_owned())
        .with_layout(facts.layout)
        .with_required_bytes(required_bytes)
        .with_limit_bytes(max_binding_bytes);
    if let Some(helper_buffer) = helper_buffer {
        blocker = blocker.with_helper_buffer(helper_buffer);
    }
    blockers.push(blocker);
}

fn direct_storage_binding_bytes(io: &BoundLogicalIo<'_>, limits: FftDeviceLimits) -> Option<u64> {
    if io.is_segmented() {
        return None;
    }
    let range = io.physical_ranges.first()?;
    let alignment = limits.min_storage_buffer_offset_alignment.max(1);
    let binding_offset = align_down(range.offset_bytes, alignment);
    let leading_bytes = range.offset_bytes.checked_sub(binding_offset)?;
    leading_bytes.checked_add(range.size_bytes)
}

fn logical_strided_stage(role: &'static str) -> &'static str {
    if role == "input" {
        "input-strided-pack"
    } else {
        "output-strided-unpack"
    }
}

fn direct_storage_stage(role: &'static str, is_contiguous: bool) -> &'static str {
    if is_contiguous {
        if role == "input" {
            "input-storage-window"
        } else {
            "output-storage-window"
        }
    } else {
        logical_strided_stage(role)
    }
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value / alignment * alignment
}

trait LogicalDiagnosticsBlockerExt {
    fn with_default_route(self, route: &str) -> Self;
}

impl LogicalDiagnosticsBlockerExt for FftBlocker {
    fn with_default_route(self, route: &str) -> Self {
        if self.route.is_some() {
            self
        } else {
            self.with_route(route.to_owned())
        }
    }
}

fn with_blocker_if_missing(diagnostics: FftDiagnostics, blocker: FftBlocker) -> FftDiagnostics {
    if diagnostics.blockers().contains(&blocker) {
        diagnostics
    } else {
        diagnostics.with_blocker(blocker)
    }
}

fn input_format_for(kind: FftTransformKind, precision: FftPrecision) -> &'static str {
    match kind {
        FftTransformKind::C2c => complex_endpoint_format_for(precision).as_str(),
        FftTransformKind::R2c => "real-f32",
        FftTransformKind::C2r => "packed-complex-f32",
    }
}

fn output_format_for(kind: FftTransformKind, precision: FftPrecision) -> &'static str {
    match kind {
        FftTransformKind::C2c => complex_endpoint_format_for(precision).as_str(),
        FftTransformKind::R2c => "packed-complex-f32",
        FftTransformKind::C2r => "real-f32",
    }
}

pub fn create_plan(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
) -> Result<FftPlan> {
    FftPlan::c2c(device, queue, config)
}

pub fn create_plan_with_diagnostics(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
) -> std::result::Result<FftPlan, FftPlanCreationError> {
    FftPlan::c2c_with_diagnostics(device, queue, config)
}

pub fn create_r2c_plan(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
) -> Result<FftPlan> {
    FftPlan::r2c(device, queue, config)
}

pub fn create_r2c_plan_with_diagnostics(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
) -> std::result::Result<FftPlan, FftPlanCreationError> {
    FftPlan::r2c_with_diagnostics(device, queue, config)
}

pub fn create_c2r_plan(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
) -> Result<FftPlan> {
    FftPlan::c2r(device, queue, config)
}

pub fn create_c2r_plan_with_diagnostics(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
) -> std::result::Result<FftPlan, FftPlanCreationError> {
    FftPlan::c2r_with_diagnostics(device, queue, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::FftTuning;

    fn tiny_limits() -> FftDeviceLimits {
        FftDeviceLimits {
            max_storage_buffer_binding_size: 128,
            max_buffer_size: 256,
            min_storage_buffer_offset_alignment: 64,
        }
    }

    #[test]
    fn active_tuning_resolves_only_effective_large_policy_limits() {
        let implicit = FftConfig::new(8);
        let explicit = FftConfig::new(8).with_tuning(FftTuning::default());
        let implicit_summary = active_tuning_summary(&implicit, 1 << 20, 1 << 24);
        let explicit_summary = active_tuning_summary(&explicit, 1 << 20, 1 << 24);

        assert_eq!(implicit_summary, explicit_summary);
        assert_eq!(
            implicit_summary
                .requested()
                .max_storage_buffer_binding_size(),
            None
        );
        assert_eq!(implicit_summary.requested().max_buffer_size(), None);
        assert_eq!(
            implicit_summary
                .effective()
                .max_storage_buffer_binding_size(),
            Some(1 << 20)
        );
        assert_eq!(
            implicit_summary.effective().max_buffer_size(),
            Some(1 << 24)
        );
        assert_eq!(
            implicit_summary.requested().workgroup_size(),
            implicit_summary.effective().workgroup_size()
        );
        assert_eq!(
            implicit_summary.requested().fused_min_convolution_length(),
            implicit_summary.effective().fused_min_convolution_length()
        );
    }

    #[test]
    fn c2c_endpoint_and_diagnostic_formats_follow_precision() {
        assert_eq!(
            input_endpoint_format_for(FftTransformKind::C2c, FftPrecision::F32),
            FftEndpointFormat::ComplexF32
        );
        assert_eq!(
            output_endpoint_format_for(FftTransformKind::C2c, FftPrecision::F64),
            FftEndpointFormat::ComplexF64
        );
        assert_eq!(
            input_format_for(FftTransformKind::C2c, FftPrecision::F64),
            "complex-f64"
        );
        assert_eq!(
            output_format_for(FftTransformKind::C2c, FftPrecision::F32),
            "complex-f32"
        );
        assert_eq!(
            complex_element_format_for(FftPrecision::F64),
            ElementFormat::ComplexF64
        );
        assert_eq!(
            complex_endpoint_format_for(FftPrecision::F64).as_str(),
            "complex-f64"
        );
        assert_eq!(
            input_endpoint_format_for(FftTransformKind::C2c, FftPrecision::Df64),
            FftEndpointFormat::ComplexDf64
        );
        assert_eq!(
            output_format_for(FftTransformKind::C2c, FftPrecision::Df64),
            "complex-df64"
        );
        assert_eq!(
            complex_element_format_for(FftPrecision::Df64),
            ElementFormat::ComplexDf64
        );
    }

    #[test]
    fn logical_strided_limit_blockers_include_staging_and_binding_limits() {
        let blockers = logical_io_limit_blockers(
            LogicalIoLimitFacts {
                role: "input",
                route: "large-chunk",
                layout: "strided",
                is_contiguous: false,
                is_segmented: false,
                logical_bytes: 512,
                physical_span_bytes: 768,
                direct_binding_bytes: Some(768),
                copy_range_bytes: &[],
            },
            tiny_limits(),
        );

        assert!(blockers.iter().any(|blocker| {
            blocker.kind == FftBlockerKind::HelperBuffer
                && blocker.stage.as_deref() == Some("input-strided-pack")
                && blocker.helper_buffer.as_deref() == Some("input-logical-stage")
                && blocker.required_bytes == Some(512)
                && blocker.limit_bytes == Some(256)
        }));
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.stage.as_deref() == Some("input-strided-pack")
                && blocker.helper_buffer.as_deref() == Some("input-logical-stage")
                && blocker.required_bytes == Some(512)
                && blocker.limit_bytes == Some(128)
        }));
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.stage.as_deref() == Some("input-strided-pack")
                && blocker.helper_buffer.is_none()
                && blocker.required_bytes == Some(768)
                && blocker.limit_bytes == Some(128)
        }));
    }

    #[test]
    fn logical_segmented_strided_limits_include_physical_stage_and_copy_range() {
        let blockers = logical_io_limit_blockers(
            LogicalIoLimitFacts {
                role: "output",
                route: "large-bridge",
                layout: "segmented-strided",
                is_contiguous: false,
                is_segmented: true,
                logical_bytes: 512,
                physical_span_bytes: 1024,
                direct_binding_bytes: None,
                copy_range_bytes: &[300],
            },
            tiny_limits(),
        );

        assert!(blockers.iter().any(|blocker| {
            blocker.kind == FftBlockerKind::HelperBuffer
                && blocker.helper_buffer.as_deref() == Some("output-physical-stage")
                && blocker.required_bytes == Some(1024)
                && blocker.limit_bytes == Some(256)
        }));
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.stage.as_deref() == Some("output-strided-unpack")
                && blocker.helper_buffer.as_deref() == Some("output-physical-stage")
                && blocker.required_bytes == Some(1024)
                && blocker.limit_bytes == Some(128)
        }));
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.stage.as_deref() == Some("output-segmented-copy-window")
                && blocker.required_bytes == Some(300)
                && blocker.limit_bytes == Some(256)
        }));
    }

    #[test]
    fn logical_usage_reason_includes_physical_segment_range() {
        let reason = logical_io_usage_reason("input", "COPY_SRC", 32, 96);
        assert!(reason.contains("input logical view segment"));
        assert!(reason.contains("physical offset 32"));
        assert!(reason.contains("size 96"));
        assert!(reason.contains("COPY_SRC"));
    }

    #[test]
    fn logical_diagnostics_default_route_preserves_specific_route() {
        let routed = FftBlocker::new(FftBlockerKind::Alignment, "misaligned").with_route("rader");
        assert_eq!(
            routed.with_default_route("mixed-radix").route.as_deref(),
            Some("rader")
        );

        let unrouted = FftBlocker::new(FftBlockerKind::Alignment, "misaligned");
        assert_eq!(
            unrouted.with_default_route("mixed-radix").route.as_deref(),
            Some("mixed-radix")
        );
    }

    #[test]
    fn graph_error_blockers_are_added_to_base_diagnostics() {
        let diagnostics = FftDiagnostics::new(FftRouteSummary::new("c2c", "axis-sequence"));
        let diagnostics = add_error_diagnostic_blockers(
            diagnostics,
            "axis-sequence",
            &FftError::LargeGraphStageUnsupported {
                stage: "axis-sequence-child",
                reason: "stage logical range must not be empty",
            },
        );

        let blocker = &diagnostics.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::Unsupported);
        assert_eq!(blocker.route.as_deref(), Some("axis-sequence"));
        assert_eq!(blocker.stage.as_deref(), Some("axis-sequence-child"));
        assert_eq!(blocker.layout.as_deref(), Some("logical range"));
        assert_eq!(blocker.required_bytes, Some(1));
        assert_eq!(blocker.actual_bytes, Some(0));
    }

    #[test]
    fn graph_error_blockers_preserve_specific_error_routes() {
        let diagnostics = FftDiagnostics::new(FftRouteSummary::new("c2c", "axis-sequence"));
        let diagnostics = add_error_diagnostic_blockers(
            diagnostics,
            "axis-sequence",
            &FftError::HelperBufferTooLarge {
                helper_buffer: "wgpu_fft.c2c.bridge.rader.work",
                requested_bytes: 2048,
                max_buffer_size: 1024,
            },
        );

        let blocker = &diagnostics.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::HelperBuffer);
        assert_eq!(blocker.route.as_deref(), Some("rader-bridge"));
        assert_eq!(
            blocker.stage.as_deref(),
            Some("large-bridge-helper-windows")
        );
        assert_eq!(
            blocker.helper_buffer.as_deref(),
            Some("wgpu_fft.c2c.bridge.rader.work")
        );
        assert_eq!(blocker.required_bytes, Some(2048));
        assert_eq!(blocker.limit_bytes, Some(1024));
    }

    #[test]
    fn fallback_workspace_blocker_preserves_workspace_context() {
        let blocker = fallback_workspace_blocker(&FftError::ZeroLength, 64, 128);

        assert_eq!(blocker.kind, FftBlockerKind::Workspace);
        assert_eq!(blocker.stage.as_deref(), Some("workspace"));
        assert_eq!(blocker.helper_buffer.as_deref(), Some("workspace"));
        assert_eq!(blocker.actual_bytes, Some(64));
        assert_eq!(blocker.required_bytes, Some(128));
    }
}
