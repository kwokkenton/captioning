import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from nocap.utils import get_device, count_trainable_params
from nocap.dataset import Flickr30k
from nocap.models import get_text_inputs_and_targets, collate_fn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

NUM_WORKERS = 1

def calculate_accuracy(scores, y, pad_token_id=None):
    # scores: B, N, N_classes
    # y: B, N
    if pad_token_id is not None:
        # Remove padding tokens from the target
        mask = y != pad_token_id
        scores = scores[mask]
        y = y[mask]
    correct = torch.sum(scores.argmax(dim=-1) == y)
    return correct/y.numel(), correct

class Validator:
    def __init__(self, validation_dataloader: DataLoader, device: torch.device):
        self.valid_dl = validation_dataloader
        self.device = device

    def validate(self, model: torch.nn.Module, loss_fn: torch.nn.Module):
        total_correct = 0
        count = 0
        total_loss = 0.

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.valid_dl)):
                x, y = batch

                # THIS IS EXACTLY THE SAME CODE AS THE FORWARD -----------------

                processed_images, processed_text, attention_mask = model.process_batch(x, y)
                text_inputs, input_attention_mask, text_targets = get_text_inputs_and_targets(processed_text, 
                                                                        attention_mask,
                                                                        model.eos_id, 
                                                                        model.pad_id)
                processed_images = processed_images.to(self.device)
                text_inputs = text_inputs.to(self.device)
                text_targets = text_targets.to(self.device)
                input_attention_mask = input_attention_mask.to(self.device)
                # Scores are (unnormalised) logits
                scores = model(processed_images, text_inputs, input_attention_mask)
                # Then do the loss
                loss = loss_fn(scores.view(text_targets.numel(), -1), text_targets.view(-1))
                #-------------------------------------------------------------------

                total_loss += loss.item()

                # SAME CODE AS IN PER BATCH EVALUATION  -----------------
                _, correct = calculate_accuracy(scores, text_targets, pad_token_id=model.pad_id)
                #-------------------------------------------------------------------
                total_correct += correct
                count += torch.numel(text_targets)

        return total_loss / len(self.valid_dl), total_correct / count

class Trainer:
    def __init__(
        self,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        setup_config: dict,
        device: torch.device,
    ):

        self.setup_config = setup_config
        self.batch_size = setup_config.get('batch_size')
        self.device = device
        self.train_ds = train_ds
        self.val_ds = val_ds
        # Set up datasets and dataloaders

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn
        )
        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn
        )

        self.validator = Validator(self.val_dl, self.device)

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        batches_print_frequency: int = 100,
    ):

        running_loss = 0.
        last_loss = 0.

        model = model.to(self.device)

        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            # Zero your gradients for every batch!
            optimiser.zero_grad()
            # y is a padded sequence
            # x shape B, D
            x, y = batch

            # CHANGE THIS-------------------------------------------------------
            # Training forward code
            processed_images, processed_text, attention_mask = model.process_batch(x, y)
            text_inputs, input_attention_mask, text_targets = get_text_inputs_and_targets(processed_text, 
                                                                    attention_mask,
                                                                    model.eos_id, 
                                                                    model.pad_id)
            processed_images = processed_images.to(self.device)
            text_inputs = text_inputs.to(self.device)
            text_targets = text_targets.to(self.device)
            input_attention_mask = input_attention_mask.to(self.device)
            # Scores are (unnormalised) logits
            scores = model(processed_images, text_inputs, input_attention_mask)

            # Then do the loss
            loss = loss_fn(scores.view(text_targets.numel(), -1), text_targets.view(-1))
            #-------------------------------------------------------------------
            loss.backward()
            optimiser.step()

            # Gather data and report
            running_loss += loss.item()
            if batch_idx % batches_print_frequency == (batches_print_frequency - 1):

                # CHANGE THIS---------------------------------------------------
                # Logs and sanity checking code
                logger.info(f'Correct seq:\t{model.text_tokenizer.decode(text_targets[0])}')
                logger.info(
                    f'Predicted seq:\t{model.text_tokenizer.decode(scores.argmax(-1)[0])}')
                
                with torch.no_grad():   
                    # Predict first token
                    generated = model.forward_sequential(processed_images[0:1])
                    logger.info(
                        f'Sequentially predicted seq:\t{model.text_tokenizer.decode(generated[0])}')
  
                # Calculate accuracy metric
                accuracy, _ = calculate_accuracy(scores, text_targets, pad_token_id=model.pad_id)
                ppl = torch.exp(loss)
                # loss per batch
                last_loss = running_loss / batches_print_frequency
                logger.info(
                    f'  For batch {batch_idx + 1}, the loss is {last_loss}, the accuracy is {accuracy}, the perplexity is {ppl}, ',
                )
                #-------------------------------------------------------------------
                running_loss = 0.
                
                # In case you want to save every printed time
                #    checkpoint = {
                #     'model_state_dict': model.state_dict(),
                #     'optimiser_state_dict': optimiser.state_dict(),
                # }
                # checkpoint_path = os.path.join(
                #         '/Users/kenton/projects/mlx-institute/transformer/checkpoints',
                #         f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                #     )
                #     # torch.save(checkpoint, checkpoint_path)  

        return last_loss, accuracy

    def train(
        self,
        epochs: int,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        config: dict,
    ):

        logger.info(f'Training config: {config}')
        log_to_wandb = config.get('log_to_wandb')
        log_locally = config.get('log_locally')
        checkpoint_folder = config.get(
            'checkpoint_folder'
        )

        if log_to_wandb:
            run = wandb.init(
                entity='kwokkenton-individual',
                project='mlx-week4-image-captioning',
                config=config,
            )

        for epoch in range(epochs):
            logger.info(f'Training: Epoch {epoch + 1} of {epochs}')
            train_loss, train_accuracy = self.train_one_epoch(
                model, loss_fn, optimiser, config.get(
                    'batches_print_frequency',
                ),
            )
            logger.info(
                f'Validating: Epoch {epoch + 1} of {epochs}.',
            )
            # Run validation to sanity check the model
            val_loss, val_accuracy = self.validator.validate(
                model, loss_fn,
            )
            logger.info(
                f'Epoch {epoch + 1} of {epochs} train loss: {train_loss}'
                f'train accuracy: {train_accuracy} val loss: {val_loss}'
                f'val accuracy: {val_accuracy}',
            )
            if log_locally or log_to_wandb:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                }

                if log_locally:
                    checkpoint_path = os.path.join(
                        checkpoint_folder,
                        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                    )
                    torch.save(checkpoint, checkpoint_path)

                if log_to_wandb:
                    wandb.log({
                        'train/loss': train_loss,
                        'train/accuracy': train_accuracy,
                        'val/loss': val_loss,
                        'val/accuracy': val_accuracy,
                    })

                    checkpoint_path = os.path.join(
                        wandb.run.dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                    )
                    torch.save(checkpoint, checkpoint_path)
                    artifact = wandb.Artifact('cnn_encoder', type='checkpoint')
                    artifact.add_file(checkpoint_path)
                    wandb.run.log_artifact(artifact)



if __name__ == '__main__':

    from nocap.models import clip_model_dict, ImageCaptioner
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(
            description="Example script with a --log_to_wandb flag"
        )
        parser.add_argument(
            "--log_to_wandb",
            action="store_true",
            help="If set, enable logging to Weights & Biases"
        )
        return parser.parse_args()
    
    args = parse_args()
    log_to_wandb = args.log_to_wandb

    # Model configs
    model_config = {
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
    }

    # Config parameters
    setup_config = {'batch_size': 32}

    # Training configs
    training_config = {
        'epochs': 5,
        'lr': 1e-4,
        'log_locally': False,
        'log_to_wandb': log_to_wandb,
        'batches_print_frequency': 100,
        'checkpoint_folder': 'checkpoints',
    }

    device = get_device()

    model = ImageCaptioner(clip_model_dict, model_config)
    
    num_params = count_trainable_params(model)
    logger.info(f'There are {num_params} trainable parameters in the model.')
    logger.info(model)

    train_ds = Flickr30k(split='train')
    val_ds = Flickr30k(split='val')
    
    optimiser = torch.optim.Adam(
        model.trainable_params(), lr=training_config.get('lr'),
    )

    # Load previously trained model
    # checkpoint_path = get_wandb_checkpoint_path(
    #     'kwokkenton-individual/mlx-week2-search-engine/towers_rnn:latest',
    # )

    # # Load the model
    # checkpoint = torch.load(
    #     '/Users/kenton/projects/mlx-institute/transformer/checkpoints/20250502_131922.pth', 
    #     map_location=device, weights_only=True,
    # )
    # model.load_state_dict(checkpoint['model_state_dict'])
    # Ignore pad id
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.pad_id)

    trainer = Trainer(
        train_ds=train_ds,
        val_ds=val_ds,
        setup_config=setup_config,
        device=device,
    )
    # trainer.train_one_epoch(
    #     model=model,
    #     loss_fn=loss_fn,
    #     optimiser=optimiser,
    #     batches_print_frequency=training_config.get('batches_print_frequency'),
    # )

    trainer.train(
        epochs=training_config.get('epochs'),
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        config=training_config,
    )
