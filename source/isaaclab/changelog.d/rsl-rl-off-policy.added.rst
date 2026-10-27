* Added observation previews that include the current sample without advancing delay, history, or tensor processor
  state, used to capture terminal observations before autoreset.
* Added :meth:`~isaaclab.utils.tensor_processors.TensorProcessor.preview`; custom stateful processors must implement
  it to be used with terminal observation capture.
* Allowed native dictionary fields to opt into dynamic configuration options while retaining fixed-schema validation
  elsewhere.
