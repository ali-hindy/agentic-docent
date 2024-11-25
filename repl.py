import os
from termcolor import colored
from pipeline import DocentPipeline
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
  dataset_dir = "./data_v3/images"
  json_dir = "./data_v3/json"
  pipeline = DocentPipeline(
    dataset_dir, 
    json_dir, 
    os.getenv('TOGETHER_API_KEY'),
    sim_threshold=0.9,
    embedding_type="CLIP"
  )

  print(colored("\nWelcome! I'm an art docent agent.\nGive me an image of a painting and I'll tell you about its history.", "cyan"))

  while True:
    path = input("Drag painting image here: ")

    if path == "":
      print("Exiting the Docent REPL. Goodbye!")
      break

    print(colored(f"\nRunning docent pipeline for image {path}...", "blue"))
    res = pipeline.run(path)
    print(colored(f"FINAL RESPONSE FOR IMAGE {path}:\n\n{res}\n", "green"))