Module rxitect.models.smiles_transformer
========================================

Classes
-------

`SmilesTransformer(net: torch.nn.modules.module.Module, max_lr: float = 0.001)`
:   Hooks to be used in LightningModule.

    ### Ancestors (in MRO)

    * pytorch_lightning.core.lightning.LightningModule
    * pytorch_lightning.core.mixins.device_dtype_mixin.DeviceDtypeModuleMixin
    * pytorch_lightning.core.mixins.hparams_mixin.HyperparametersMixin
    * pytorch_lightning.core.saving.ModelIO
    * pytorch_lightning.core.hooks.ModelHooks
    * pytorch_lightning.core.hooks.DataHooks
    * pytorch_lightning.core.hooks.CheckpointHooks
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `configure_optimizers(self)`
    :   Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        
        Return:
            Any of these 6 options.
        
            - **Single optimizer**.
            - **List or Tuple** of optimizers.
            - **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers
              (or multiple ``lr_scheduler_config``).
            - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
              key whose value is a single LR scheduler or ``lr_scheduler_config``.
            - **Tuple of dictionaries** as described above, with an optional ``"frequency"`` key.
            - **None** - Fit will run without any optimizer.
        
        The ``lr_scheduler_config`` is a dictionary which contains the scheduler and its associated configuration.
        The default configuration is shown below.
        
        .. code-block:: python
        
            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }
        
        When there are schedulers in which the ``.step()`` method is conditioned on a value, such as the
        :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler, Lightning requires that the
        ``lr_scheduler_config`` contains the keyword ``"monitor"`` set to the metric name that the scheduler
        should be conditioned on.
        
        .. testcode::
        
            # The ReduceLROnPlateau scheduler requires a monitor
            def configure_optimizers(self):
                optimizer = Adam(...)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(optimizer, ...),
                        "monitor": "metric_to_track",
                        "frequency": "indicates how often the metric is updated"
                        # If "monitor" references validation metrics, then "frequency" should be set to a
                        # multiple of "trainer.check_val_every_n_epoch".
                    },
                }
        
        
            # In the case of two optimizers, only one using the ReduceLROnPlateau scheduler
            def configure_optimizers(self):
                optimizer1 = Adam(...)
                optimizer2 = SGD(...)
                scheduler1 = ReduceLROnPlateau(optimizer1, ...)
                scheduler2 = LambdaLR(optimizer2, ...)
                return (
                    {
                        "optimizer": optimizer1,
                        "lr_scheduler": {
                            "scheduler": scheduler1,
                            "monitor": "metric_to_track",
                        },
                    },
                    {"optimizer": optimizer2, "lr_scheduler": scheduler2},
                )
        
        Metrics can be made available to monitor by simply logging it using
        ``self.log('metric_to_track', metric_val)`` in your :class:`~pytorch_lightning.core.lightning.LightningModule`.
        
        Note:
            The ``frequency`` value specified in a dict along with the ``optimizer`` key is an int corresponding
            to the number of sequential batches optimized with the specific optimizer.
            It should be given to none or to all of the optimizers.
            There is a difference between passing multiple optimizers in a list,
            and passing multiple optimizers in dictionaries with a frequency of 1:
        
                - In the former case, all optimizers will operate on the given batch in each optimization step.
                - In the latter, only one optimizer will operate on the given batch at every step.
        
            This is different from the ``frequency`` value specified in the ``lr_scheduler_config`` mentioned above.
        
            .. code-block:: python
        
                def configure_optimizers(self):
                    optimizer_one = torch.optim.SGD(self.model.parameters(), lr=0.01)
                    optimizer_two = torch.optim.SGD(self.model.parameters(), lr=0.01)
                    return [
                        {"optimizer": optimizer_one, "frequency": 5},
                        {"optimizer": optimizer_two, "frequency": 10},
                    ]
        
            In this example, the first optimizer will be used for the first 5 steps,
            the second optimizer for the next 10 steps and that cycle will continue.
            If an LR scheduler is specified for an optimizer using the ``lr_scheduler`` key in the above dict,
            the scheduler will only be updated when its optimizer is being used.
        
        Examples::
        
            # most cases. no learning rate scheduler
            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-3)
        
            # multiple optimizer case (e.g.: GAN)
            def configure_optimizers(self):
                gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
                return gen_opt, dis_opt
        
            # example with learning rate schedulers
            def configure_optimizers(self):
                gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
                dis_sch = CosineAnnealing(dis_opt, T_max=10)
                return [gen_opt, dis_opt], [dis_sch]
        
            # example with step-based learning rate schedulers
            # each optimizer has its own scheduler
            def configure_optimizers(self):
                gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
                gen_sch = {
                    'scheduler': ExponentialLR(gen_opt, 0.99),
                    'interval': 'step'  # called after each training step
                }
                dis_sch = CosineAnnealing(dis_opt, T_max=10) # called every epoch
                return [gen_opt, dis_opt], [gen_sch, dis_sch]
        
            # example with optimizer frequencies
            # see training procedure in `Improved Training of Wasserstein GANs`, Algorithm 1
            # https://arxiv.org/abs/1704.00028
            def configure_optimizers(self):
                gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
                n_critic = 5
                return (
                    {'optimizer': dis_opt, 'frequency': n_critic},
                    {'optimizer': gen_opt, 'frequency': 1}
                )
        
        Note:
            Some things to know:
        
            - Lightning calls ``.backward()`` and ``.step()`` on each optimizer and learning rate scheduler as needed.
            - If you use 16-bit precision (``precision=16``), Lightning will automatically handle the optimizers.
            - If you use multiple optimizers, :meth:`training_step` will have an additional ``optimizer_idx`` parameter.
            - If you use :class:`torch.optim.LBFGS`, Lightning handles the closure function automatically for you.
            - If you use multiple optimizers, gradients will be calculated only for the parameters of current optimizer
              at each training step.
            - If you need to control how often those optimizers step or override the default ``.step()`` schedule,
              override the :meth:`optimizer_step` hook.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Same as :meth:`torch.nn.Module.forward()`.
        
        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.
        
        Return:
            Your model's output

    `step(self, batch: torch.Tensor)`
    :

    `test_step(self, batch: torch.Tensor, batch_idx: int) ‑> float`
    :   Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest
        such as accuracy.
        
        .. code-block:: python
        
            # the pseudocode for these calls
            test_outs = []
            for test_batch in test_data:
                out = test_step(test_batch)
                test_outs.append(out)
            test_epoch_end(test_outs)
        
        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_id: The index of the dataloader that produced this batch.
                (only if multiple test dataloaders used).
        
        Return:
           Any of.
        
            - Any object or value
            - ``None`` - Testing will skip to the next batch
        
        .. code-block:: python
        
            # if you have one test dataloader:
            def test_step(self, batch, batch_idx):
                ...
        
        
            # if you have multiple test dataloaders:
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                ...
        
        Examples::
        
            # CASE 1: A single test dataset
            def test_step(self, batch, batch_idx):
                x, y = batch
        
                # implement your own
                out = self(x)
                loss = self.loss(out, y)
        
                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)
        
                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        
                # log the outputs!
                self.log_dict({'test_loss': loss, 'test_acc': test_acc})
        
        If you pass in multiple test dataloaders, :meth:`test_step` will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.
        
        .. code-block:: python
        
            # CASE 2: multiple test dataloaders
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...
        
        Note:
            If you don't need to test you don't need to implement this method.
        
        Note:
            When the :meth:`test_step` is called, the model has been put in eval mode and
            PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
            to training mode and gradients are enabled.

    `training_step(self, batch: torch.Tensor, batch_idx: int) ‑> float`
    :   Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        
        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (``int``): Integer displaying index of this batch
            optimizer_idx (``int``): When using multiple optimizers, this argument will also be present.
            hiddens (``Any``): Passed in if
                :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` > 0.
        
        Return:
            Any of.
        
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``
            - ``None`` - Training will skip to the next batch. This is only for automatic optimization.
                This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.
        
        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.
        
        Example::
        
            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss
        
        If you define multiple optimizers, this step will be called with an additional
        ``optimizer_idx`` parameter.
        
        .. code-block:: python
        
            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx, optimizer_idx):
                if optimizer_idx == 0:
                    # do training_step with encoder
                    ...
                if optimizer_idx == 1:
                    # do training_step with decoder
                    ...
        
        
        If you add truncated back propagation through time you will also get an additional
        argument with the hidden states of the previous step.
        
        .. code-block:: python
        
            # Truncated back-propagation through time
            def training_step(self, batch, batch_idx, hiddens):
                # hiddens are the hidden states from the previous truncated backprop step
                out, hiddens = self.lstm(data, hiddens)
                loss = ...
                return {"loss": loss, "hiddens": hiddens}
        
        Note:
            The loss value shown in the progress bar is smoothed (averaged) over the last values,
            so it differs from the actual loss returned in train/validation step.

    `validation_step(self, batch: torch.Tensor, batch_idx: int) ‑> float`
    :   Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        
        .. code-block:: python
        
            # the pseudocode for these calls
            val_outs = []
            for val_batch in val_data:
                out = validation_step(val_batch)
                val_outs.append(out)
            validation_epoch_end(val_outs)
        
        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple val dataloaders used)
        
        Return:
            - Any object or value
            - ``None`` - Validation will skip to the next batch
        
        .. code-block:: python
        
            # pseudocode of order
            val_outs = []
            for val_batch in val_data:
                out = validation_step(val_batch)
                if defined("validation_step_end"):
                    out = validation_step_end(out)
                val_outs.append(out)
            val_outs = validation_epoch_end(val_outs)
        
        
        .. code-block:: python
        
            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx):
                ...
        
        
            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                ...
        
        Examples::
        
            # CASE 1: A single validation dataset
            def validation_step(self, batch, batch_idx):
                x, y = batch
        
                # implement your own
                out = self(x)
                loss = self.loss(out, y)
        
                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)
        
                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        
                # log the outputs!
                self.log_dict({'val_loss': loss, 'val_acc': val_acc})
        
        If you pass in multiple val dataloaders, :meth:`validation_step` will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.
        
        .. code-block:: python
        
            # CASE 2: multiple validation dataloaders
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...
        
        Note:
            If you don't need to validate you don't need to implement this method.
        
        Note:
            When the :meth:`validation_step` is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of validation,
            the model goes back to training mode and gradients are enabled.