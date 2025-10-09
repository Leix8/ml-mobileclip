import torch
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model
from prettytable import PrettyTable
import os
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import argparse  # Import argparse for command line argument parsing

def print_result_table(image, scores, probs, labels, output_dir):  # Added output_dir parameter
    table = PrettyTable()
    table.field_names = ["Label", "Score"]
    for label, score, prob in zip(labels, scores[0].tolist(), probs[0].tolist()):
        table.add_row([label, f"{score:.4f}"])
    print(table)

    # Save scores to a file in the specified output directory
    with open(os.path.join(output_dir, "scores_output.txt"), "a") as f:  # Append mode
        f.write(f"Scores for {image}:\n")
        f.write(table.get_string() + "\n")  # Write the PrettyTable format to the file
        f.write("\n")

    # Plotting the scores as a bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores[0].tolist(), color='skyblue')  # Change to bar plot
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.xlabel('Labels')
    plt.ylabel('Scores')
    plt.title(f'Scores for {image}')
    plt.ylim(-0.3, 0.7)  # Set y-axis limits to static range -1 to 1
    plt.tight_layout()  # Adjust layout to make sure everything fits
    plt.savefig(os.path.join(output_dir, f"{image}_scores.png"))  # Save the plot as an image in the output directory
    plt.close()  # Close the plot to free memory

def main(image_dir, output_dir):
    text_labels = []
    text_labels += ["a dog", " a dog is playing", "a dog is playing on the grass", "a dog is playing on the grass with a soccer"]
    text_labels += ["a dog", " a dog is running", "a dog is running on the grass"]
    text_labels += ["a dog", " a dog is running", "a dog is running in the living room"]
    text_labels += ["a dog", " a dog is eating", "a dog is eating bones", "a dog is eating bones in the bedroom"]
    text_labels += ["a dog", " a dog is sleeping", "a dog is sleeping in the garden"]
    text_labels +=["a dog", " a dog is playing", "a dog is playing in the bedroom"]
    text_labels += ["a cat", "a cat is jumping", "a cat is jumping in the living room", "a cat is jumping in the living room for a wand"]

    model_name = "MobileCLIP2-S4"
    model_kwargs = {}
    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="./checkpoints/mobileclip2_s4.pt", **model_kwargs)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Model needs to be in eval mode for inference because of batchnorm layers unlike ViTs
    model.eval()

    # For inference/model exporting purposes, please reparameterize first
    model = reparameterize_model(model)

    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add more image formats if needed
            image_path = os.path.join(image_dir, filename)
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)

            text = tokenizer(text_labels)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                print(f"check embeddings for {filename}: image_embedding = {image_features.shape}, text_embedding = {text_features.shape}")
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                text_scores = image_features @ text_features.T

            print_result_table(filename, text_scores, text_probs, text_labels, output_dir)  # Pass output_dir for saving results
            print(f"label scores for {filename}: {text_scores}") 
            print("Label probs:", text_probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory of input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output results')
    args = parser.parse_args()

    main(args.image_dir, args.output_dir)




