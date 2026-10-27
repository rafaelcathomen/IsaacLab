* Fixed terminal observations (``extras["final_obs"]``) advancing observation histories, delay buffers, and stateful
  tensor processors of the surviving environments, and cleared stale terminal observations between steps.
* Preserved task-owned camera histories, composite observation containers, and the integrator output dtype in
  terminal observation previews of manager-based and direct environments.
* Copied direct environment terminal observations before autoreset could overwrite task-owned tensors.
