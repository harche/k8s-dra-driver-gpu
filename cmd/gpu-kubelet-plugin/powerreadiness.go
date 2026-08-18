/*
Copyright The Kubernetes Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	resourcelisters "k8s.io/client-go/listers/resource/v1"
	"k8s.io/klog/v2"
)

// Power readiness satisfies one of the binding conditions this driver publishes
// (see bindingconditions.go) from the node itself, by checking which workload
// power profiles a GPU is actually running under.
//
// This mirrors what NVIDIA DPS already does on Slurm. A job carries its power
// settings in the job comment, including dps_wpps with the workload power
// profile IDs, and a prolog runs dpsctl to create a resource group from the
// nodes the scheduler allocated and apply those settings before the job starts.
// Kubernetes has no prolog: the only hook between choosing a node and starting
// the workload is NodePrepareResources, which runs after the pod is already
// bound. Binding conditions are that missing hook, one phase earlier, and this
// writer is the part that waits.
//
// Readiness is taken from the enforced mask, which NVML documents as the
// "mask of currently enforced performance profiles post all arbitrations among
// the requested profiles". A profile that was requested but lost an arbitration
// is not active on the GPU, so requested is not good enough to start a workload
// on.
//
// What this does not do is apply anything. Applying profiles is the power
// manager's job, via dpsctl on Slurm today. If nothing applies them on a
// Kubernetes cluster, the condition never becomes true and pods requesting
// these GPUs wait out the scheduler's binding timeout, so this is only useful
// alongside something that does.
//
// Only the readiness condition is ever written. Contention in these systems
// arbitrates between profiles rather than failing outright, and a GPU running a
// different profile is still a working GPU, so there is no honest way to turn
// that into a binding failure from here.
//
// Profiles are reported only by hardware that supports them, Blackwell class
// and later. A GPU that cannot report them is treated as ready and logged once:
// the driver has no basis for holding a workload back on a GPU it cannot ask.
type powerReadinessConfig struct {
	// profileIDs are the workload power profile IDs that must be enforced on
	// every allocated GPU before a pod may bind. Empty disables the mechanism.
	profileIDs []int
	// interval is how often claims are re-evaluated. The enforced mask is a
	// polled value, not an event, so this is the resolution of the whole
	// mechanism.
	interval time.Duration
}

// Workload power profile IDs are numbered differently depending on who is
// asking. DPS configures them out of band over Redfish and documents its range
// as 3 to 258, while NVML and nvidia-smi use a value 3 lower, which is also the
// bit index into nvmlMask255_t:
//
//	OOB value = NVML value + 3
//
// The IDs configured here are the DPS ones, so that they match the
// dpsctl --workload-profile-ids invocation that applied them, and are converted
// before the mask is indexed. Getting this wrong is invisible: the condition
// simply never becomes true and pods wait out the binding timeout.
const (
	oobProfileIDOffset        = 3
	minWorkloadPowerProfileID = oobProfileIDOffset
	// The mask holds 255 bits, so the highest usable DPS ID is 254 + 3.
	maxWorkloadPowerProfileID = 254 + oobProfileIDOffset
	profileMaskBits           = 255
)

// The condition types this driver satisfies itself. They are owned by the
// driver rather than configured: an operator who wants the driver to gate on
// power says only how many watts, and naming the condition twice (once to
// publish it, once to satisfy it) would be a way to get it wrong.
//
// A failure condition is published because the API server rejects a device
// carrying one list without the other, but this driver never sets it: a GPU
// clamped to a lower cap is still a working GPU.
const (
	powerReadyConditionType  = DriverName + "/power-ready"
	powerFailedConditionType = DriverName + "/power-failed"
)

// defaultPowerReadinessInterval matches the scheduler's own poll period while
// it waits in PreBind, so readiness is normally observed within one scheduler
// tick without polling NVML more often than the result can be consumed.
const defaultPowerReadinessInterval = 5 * time.Second

func (c powerReadinessConfig) enabled() bool {
	return len(c.profileIDs) > 0
}

func (c powerReadinessConfig) validate() error {
	var errs []error
	seen := make(map[int]bool, len(c.profileIDs))
	for _, id := range c.profileIDs {
		if id < minWorkloadPowerProfileID || id > maxWorkloadPowerProfileID {
			errs = append(errs, fmt.Errorf("--power-readiness-profile-ids: %d is out of range, workload power profile IDs are %d to %d, the numbering DPS uses and three higher than the nvidia-smi value", id, minWorkloadPowerProfileID, maxWorkloadPowerProfileID))
		}
		if seen[id] {
			errs = append(errs, fmt.Errorf("--power-readiness-profile-ids: duplicate profile ID %d", id))
		}
		seen[id] = true
	}
	return errors.Join(errs...)
}

// parseProfileIDs reads a comma separated list of workload power profile IDs.
func parseProfileIDs(s string) ([]int, error) {
	var out []int
	for _, item := range strings.Split(s, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		id, err := strconv.Atoi(item)
		if err != nil {
			return nil, fmt.Errorf("--power-readiness-profile-ids: %q is not a number", item)
		}
		out = append(out, id)
	}
	return out, nil
}

// maskHasProfile reports whether a DPS workload power profile ID is set in an
// nvmlMask255_t, converting from the DPS numbering to the NVML bit index.
func maskHasProfile(mask nvml.Mask255, oobID int) bool {
	bit := oobID - oobProfileIDOffset
	if bit < 0 || bit >= profileMaskBits {
		return false
	}
	return mask.Mask[bit/32]&(1<<uint(bit%32)) != 0
}

// powerReadinessWriter marks the configured binding condition True on claims
// allocated to this node once every allocated GPU reports a power limit at or
// above the configured floor.
//
// It has to watch the API rather than react to a kubelet call: binding
// conditions are evaluated before the pod is bound, so NodePrepareResources has
// not been called and will not be called until this writer unblocks it.
//
// That has two costs worth understanding before enabling this at scale:
//
//   - There is no field selector for "claims allocated to my node", so each
//     node watches every ResourceClaim in the cluster. Watch traffic and
//     per-node cache both grow with the number of claims. A central controller
//     would not have this problem, which is why the reference implementation
//     puts its condition writer in one; a node-local writer is only used here
//     because the readiness signal is node-local.
//   - Writing claim status requires resourceclaims/status cluster-wide. On
//     Kubernetes 1.36+ the resourceclaims/driver rule narrows that to claims
//     allocated to this node, but on older clusters that narrowing does not
//     exist and a compromised node could write any claim's device status.
//
// The claims informer is pinned to resource.k8s.io/v1, which is GA from
// Kubernetes 1.34 and therefore present on every release that can serve the
// binding condition fields this writer depends on.
type powerReadinessWriter struct {
	config *Config
	state  *DeviceState
	prc    powerReadinessConfig

	lister resourcelisters.ResourceClaimLister

	// unqueryable remembers devices whose profiles NVML cannot report, so that
	// it is logged once rather than every interval.
	unqueryableOnce sync.Map

	// readCurrentProfiles reads the profile masks for one GPU. It exists as a
	// field so that tests can exercise the readiness logic without NVML, which
	// matters here more than usual: workload power profiles are reported only
	// by Blackwell class hardware and later, so this cannot be run against a
	// GPU on most machines.
	readCurrentProfiles func(uuid string) (nvml.WorkloadPowerProfileCurrentProfiles, nvml.Return)
}

func newPowerReadinessWriter(config *Config, state *DeviceState) *powerReadinessWriter {
	w := &powerReadinessWriter{config: config, state: state, prc: config.powerReadiness}
	w.readCurrentProfiles = w.nvmlCurrentProfiles
	return w
}

// nvmlCurrentProfiles is the real reader, asking NVML for the profile masks of
// the GPU with the given UUID.
//
// Must be called with the DeviceState lock held: DeviceGetHandleByUUID writes
// an unsynchronized handle cache that prepare and unprepare also write.
func (w *powerReadinessWriter) nvmlCurrentProfiles(uuid string) (nvml.WorkloadPowerProfileCurrentProfiles, nvml.Return) {
	handle, ret := w.state.nvdevlib.DeviceGetHandleByUUID(uuid)
	if ret != nvml.SUCCESS {
		return nvml.WorkloadPowerProfileCurrentProfiles{}, ret
	}
	return handle.WorkloadPowerProfileGetCurrentProfiles()
}

// Start brings up everything that can fail, so that a driver which cannot
// satisfy the condition it is about to publish refuses to start at all. The
// alternative is worse than a crash: the node would advertise
// gpu.nvidia.com/power-ready with nothing able to set it, and every GPU pod
// scheduled there would wait out the scheduler's binding timeout and be
// rescheduled, forever, with no self-healing.
//
// The returned function runs the poll loop until ctx is canceled.
func (w *powerReadinessWriter) Start(ctx context.Context) (func(), error) {
	// NVML is reference counted and is only initialized for the duration of the
	// operations that need it, so a long-lived reader has to hold its own
	// reference. Without this every power read fails with ERROR_UNINITIALIZED.
	//
	// Use the driver's own wrapper rather than nvmllib.Init(): it passes
	// INIT_FLAG_NO_GPUS, so a node whose GPUs are all bound to vfio-pci still
	// initializes instead of failing and leaving the published condition with
	// nobody to satisfy it.
	if w.state.nvdevlib == nil || w.state.nvdevlib.nvmllib == nil {
		return nil, errors.New("nvml library is nil")
	}
	if err := w.state.nvdevlib.Init(); err != nil {
		return nil, fmt.Errorf("initializing NVML: %w", err)
	}

	factory := informers.NewSharedInformerFactory(w.config.clientsets.Core, 10*time.Minute)
	informer := factory.Resource().V1().ResourceClaims()
	w.lister = informer.Lister()

	factory.Start(ctx.Done())
	for typ, ok := range factory.WaitForCacheSync(ctx.Done()) {
		if !ok {
			w.state.nvdevlib.alwaysShutdown()
			return nil, fmt.Errorf("failed to sync informer cache for %v", typ)
		}
	}

	klog.Infof("Power readiness writer started: will set %q on claims allocated to this node once workload power profiles %v are enforced on every allocated GPU (checked every %s)",
		powerReadyConditionType, w.prc.profileIDs, w.prc.interval)

	return func() {
		defer w.state.nvdevlib.alwaysShutdown()
		ticker := time.NewTicker(w.prc.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				w.reconcileAll(ctx)
			}
		}
	}, nil
}

func (w *powerReadinessWriter) reconcileAll(ctx context.Context) {
	claims, err := w.lister.List(labels.Everything())
	if err != nil {
		klog.Errorf("Power readiness: listing ResourceClaims failed: %v", err)
		return
	}
	for _, claim := range claims {
		if err := w.reconcile(ctx, claim); err != nil {
			klog.Errorf("Power readiness: claim %s/%s: %v", claim.Namespace, claim.Name, err)
		}
	}
}

// devicesToGate returns the allocation results on this node that carry the
// condition this writer satisfies and do not have it set to True yet.
func (w *powerReadinessWriter) devicesToGate(claim *resourceapi.ResourceClaim) []resourceapi.DeviceRequestAllocationResult {
	if claim.Status.Allocation == nil {
		return nil
	}
	var out []resourceapi.DeviceRequestAllocationResult
	for _, result := range claim.Status.Allocation.Devices.Results {
		if result.Driver != DriverName || result.Pool != w.config.flags.nodeName {
			continue
		}
		if !slices.Contains(result.BindingConditions, powerReadyConditionType) {
			continue
		}
		if conditionIsTrue(claim, result, powerReadyConditionType) {
			continue
		}
		out = append(out, result)
	}
	return out
}

func (w *powerReadinessWriter) reconcile(ctx context.Context, claim *resourceapi.ResourceClaim) error {
	results := w.devicesToGate(claim)
	if len(results) == 0 {
		return nil
	}

	for _, result := range results {
		ready, err := w.deviceHasProfiles(result.Device)
		if err != nil {
			return err
		}
		if !ready {
			klog.V(4).Infof("Power readiness: claim %s/%s device %s not ready yet, workload power profiles %v are not all enforced",
				claim.Namespace, claim.Name, result.Device, w.prc.profileIDs)
			return nil
		}
	}

	// Every gated device on this node is at or above the floor.
	updated := claim.DeepCopy()
	now := metav1.Now()
	for _, result := range results {
		setDeviceCondition(updated, result, metav1.Condition{
			Type:               powerReadyConditionType,
			Status:             metav1.ConditionTrue,
			Reason:             "WorkloadPowerProfilesEnforced",
			Message:            fmt.Sprintf("workload power profiles %v are enforced", w.prc.profileIDs),
			LastTransitionTime: now,
		})
	}

	_, err := w.config.clientsets.Resource.ResourceClaims(updated.Namespace).UpdateStatus(ctx, updated, metav1.UpdateOptions{})
	switch {
	case apierrors.IsConflict(err):
		// Someone else wrote the claim first; the next tick works from a
		// refreshed cache.
		klog.V(5).Infof("Power readiness: conflict updating claim %s/%s, retrying next interval", updated.Namespace, updated.Name)
		return nil
	case apierrors.IsNotFound(err):
		return nil
	case err != nil:
		return fmt.Errorf("updating status: %w", err)
	}

	klog.Infof("Power readiness: set %q on claim %s/%s for %d device(s)",
		powerReadyConditionType, updated.Namespace, updated.Name, len(results))
	return nil
}

// deviceHasProfiles reports whether every configured workload power profile is
// currently enforced on the GPU backing the named device.
//
// A device whose profiles NVML cannot report (a GPU bound to vfio-pci, or
// hardware older than Blackwell) is treated as ready: the driver has no basis
// for holding a workload back on a GPU it cannot ask, and blocking would strand
// it until the scheduler's binding timeout.
func (w *powerReadinessWriter) deviceHasProfiles(deviceName string) (bool, error) {
	// Both the allocatable device map and the NVML handle cache inside
	// DeviceGetHandleByUUID are written by prepare/unprepare while they hold
	// the DeviceState lock. Reading them from this goroutine without it is a
	// concurrent map access, which is fatal in Go rather than merely racy. The
	// read below is an NVML query measured in microseconds, so holding the lock
	// across it does not meaningfully delay a prepare.
	w.state.Lock()
	defer w.state.Unlock()

	uuid, ok := w.gpuUUIDForDevice(DeviceName(deviceName))
	if !ok {
		w.logUnqueryableOnce(deviceName, "no GPU UUID could be resolved for it")
		return true, nil
	}

	current, ret := w.readCurrentProfiles(uuid)
	if ret == nvml.ERROR_NOT_SUPPORTED {
		w.logUnqueryableOnce(deviceName, "NVML reports workload power profiles as unsupported on this GPU")
		return true, nil
	}
	if ret != nvml.SUCCESS {
		return false, fmt.Errorf("reading workload power profiles for %s: %v", uuid, ret)
	}

	for _, id := range w.prc.profileIDs {
		if !maskHasProfile(current.EnforcedProfilesMask, id) {
			return false, nil
		}
	}
	return true, nil
}

// gpuUUIDForDevice maps a published device name to the UUID of the physical GPU
// whose power limit governs it. MIG partitions do not have their own power
// budget, so they resolve to their parent.
//
// Must be called with the DeviceState lock held.
func (w *powerReadinessWriter) gpuUUIDForDevice(name DeviceName) (string, bool) {
	for _, devices := range w.state.perGPUAllocatable.allocatablesMap {
		device, ok := devices[name]
		if !ok {
			continue
		}
		switch {
		case device.Gpu != nil:
			return device.Gpu.UUID, true
		case device.MigStatic != nil:
			return device.MigStatic.ParentUUID, true
		case device.MigDynamic != nil && device.MigDynamic.Parent != nil:
			return device.MigDynamic.Parent.UUID, true
		}
		// A VFIO device is bound to vfio-pci and is not visible to NVML.
		return "", false
	}
	return "", false
}

func (w *powerReadinessWriter) logUnqueryableOnce(deviceName, why string) {
	if _, seen := w.unqueryableOnce.LoadOrStore(deviceName, struct{}{}); seen {
		return
	}
	klog.Warningf("Power readiness: treating device %s as ready because %s; %q will be satisfied for it without a power check",
		deviceName, why, powerReadyConditionType)
}

// conditionIsTrue reports whether the claim already records the condition as
// True for the given device.
func conditionIsTrue(claim *resourceapi.ResourceClaim, result resourceapi.DeviceRequestAllocationResult, conditionType string) bool {
	for _, status := range claim.Status.Devices {
		if status.Driver != result.Driver || status.Pool != result.Pool || status.Device != result.Device {
			continue
		}
		for _, condition := range status.Conditions {
			if condition.Type == conditionType {
				return condition.Status == metav1.ConditionTrue
			}
		}
	}
	return false
}

// setDeviceCondition records a condition for one device, leaving every other
// device entry -- including those written by other drivers -- untouched.
func setDeviceCondition(claim *resourceapi.ResourceClaim, result resourceapi.DeviceRequestAllocationResult, condition metav1.Condition) {
	for i := range claim.Status.Devices {
		status := &claim.Status.Devices[i]
		if status.Driver != result.Driver || status.Pool != result.Pool || status.Device != result.Device {
			continue
		}
		for j := range status.Conditions {
			if status.Conditions[j].Type == condition.Type {
				if status.Conditions[j].Status != condition.Status {
					status.Conditions[j] = condition
				}
				return
			}
		}
		status.Conditions = append(status.Conditions, condition)
		return
	}
	status := resourceapi.AllocatedDeviceStatus{
		Driver:     result.Driver,
		Pool:       result.Pool,
		Device:     result.Device,
		Conditions: []metav1.Condition{condition},
	}
	if result.ShareID != nil {
		shareID := string(*result.ShareID)
		status.ShareID = &shareID
	}
	claim.Status.Devices = append(claim.Status.Devices, status)
}
