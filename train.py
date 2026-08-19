import argparse
import itertools
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from utils.utils import *
from utils.models import *
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--content_dir", type=str, default = "/Users/subash_lamichhane/Documents/Major_Projects/Neural_Style_Transfer/data/content_data",
                        help="Location of dataset")
    
    parser.add_argument("--style_dir", type=str, default = "/Users/subash_lamichhane/Documents/Major_Projects/Neural_Style_Transfer/data/style_data",
                        help="Location of style dataset")
    
    parser.add_argument("--vgg", type=str, default="/Users/subash_lamichhane/Documents/Major_Projects/Neural_Style_Transfer/vgg_normalised.pth",
                        help="Location of pre-trained vgg")
    
    parser.add_argument("--experiment", type=str, default="experiment1", help="Name of experiment")
    
    parser.add_argument("--final_size", type=int, default=256, help="Final size of image")
    
    parser.add_argument("--content_size", type=int, default=256, help="Size of content image")
    
    parser.add_argument("--style_size", type=int, default=256, help="Size of style image")
    
    parser.add_argument("--crop", action="store_true", default=True, help="Crop the image")
    
    parser.add_argument("--batch_size", type=int,default=4, help="Batch size for training")
    
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    
    parser.add_argument("--lr_decay", type=float, default=5e-5, help="Learning rate decay for training")
    
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    
    parser.add_argument("--content_weight", type=float, default=1.0, help="Weight for content loss")
    
    parser.add_argument("--style_weight", type=float, default=5.0, help="Weight for style loss")
    
    parser.add_argument("--log_interval", type=int, default=1, help="Interval for logging training progress")
    
    parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving model checkpoints")
    
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint")
    
    parser.add_argument("--decoder_path", type=str, default=None, help="Path to the decoder checkpoint")
    
    parser.add_argument("--optimizer_path", type=str, default=None, help="Path to the optimizer checkpoint")
    
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
        
    save_dir = Path(f"experiments")/args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save argument values
    with open(save_dir / "args.txt", "w") as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key}: {value}\n")
            
            
    content_transform=get_transform(args.content_size, args.crop, args.final_size)
    style_transform=get_transform(args.style_size, args.crop, args.final_size)
    
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)
    
    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda = lambda epoch : 1.0 / (1.0 + args.lr_decay * epoch)
    )
    
    if args.resume:
        decoder_checkpoint = torch.load(args.decoder_path, map_location=device)
        decoder.load_state_dict(decoder_checkpoint)
        
        optimizer_checkpoint = torch.load(args.optimizer_path, map_location=device)
        optimizer.load_state_dict(optimizer_checkpoint)
        
        print("Resumed training from the latest checkpoint.")
        
        
        
        
    print("Training started...")
    
    mse_loss = torch.nn.MSELoss()
    encoder.eval()  # Set encoder to evaluation mode
    
    running_loss = None
    running_content_loss = None
    running_style_loss = None
    
    
    for epoch in range(args.epochs):
        # style_data is much smaller than content_data, so cycle it. Without this,
        # zip() truncates every epoch to len(style_dataloader) batches and a large
        # chunk of content_data is silently never used.
        progress_bar = tqdm(zip(content_dataloader, itertools.cycle(style_dataloader)), total=len(content_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")

        running_loss = 0.0
        running_content_loss = 0.0
        running_style_loss = 0.0
        
        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)
            
            t= adaptive_instance_normalization(c_feats[-1], s_feats[-1])
           
            generated_output = decoder(t)
           
            g_feats = encoder(generated_output)
            
            content_loss = mse_loss(g_feats[-1], t)* args.content_weight
            style_loss = 0.0
            
            for g_f, s_f in zip(g_feats, s_feats):
                
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                
                style_loss += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
                
            style_loss *= args.style_weight
            total_loss = content_loss + style_loss
           
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss.item():.4f}, Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}")

            running_loss += total_loss.item()
            running_content_loss += content_loss.item()
            running_style_loss += style_loss.item()

        scheduler.step()
        
        avg_loss = running_loss / len(content_dataloader)
        avg_content_loss = running_content_loss / len(content_dataloader)
        avg_style_loss = running_style_loss / len(content_dataloader)
        
        
        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(f"Epoch [{epoch+1}/{args.epochs}], Avg Loss: {avg_loss:.4f}, Avg Content Loss: {avg_content_loss:.4f}, Avg Style Loss: {avg_style_loss:.4f}")

        if (epoch +1)% args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f"decoder_epoch_{epoch+1}.pth")
            print(f"Saved model at epoch {epoch+1}")
            torch.save(optimizer.state_dict(), save_dir / f"optimizer_epoch_{epoch+1}.pth")
            
            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, generated_output], dim=0)
                save_image(output, save_dir / f"output_epoch_{epoch+1}.png", nrow=args.batch_size)
                


if __name__ == "__main__":
    main()