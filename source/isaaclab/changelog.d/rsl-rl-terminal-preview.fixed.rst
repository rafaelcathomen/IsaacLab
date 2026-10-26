* Preserved built-in observation modifier state during terminal previews and required custom class-based modifiers to provide a non-mutating preview.
* Copied direct environment terminal observations before autoreset could overwrite task-owned tensors.
* Preserved composite observation containers and the live integrator output dtype in terminal previews.
