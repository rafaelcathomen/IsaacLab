* Custom stateful tensor processors should override ``preview`` to preserve their state during terminal captures;
  the default delegates to ``__call__`` for compatibility. ``ObservationManager`` subclasses overriding
  ``compute_group`` must accept the new keyword-only ``preview`` argument.
