import json
import pandas as pd
import torch

from datasets import Dataset, Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM

from pprint import pformat
import logging
from .common import init_logging
from .common import parse_general_args


class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(
            images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")

        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        return encoding


def create_dataset(json_file):
    image_folder = 'aux_data/raw_data/'
    # json_file = 'aux_data/raw_data/coco_karpathy_train_indo.json'

    # captions = json.loads(read_to_buffer(json_file))
    f = open(json_file)
    captions = json.load(f)
    f.close()

    df = pd.DataFrame(captions)
    df['image'] = df.image.map(lambda image_path: image_folder + image_path)

    images = df['image'].tolist()
    captions = df['caption'].tolist()

    dataset = Dataset.from_dict(
        {"image": images, "text": captions}).cast_column("image", Image())
    return dataset


def finetune_git(json_file, model_id, out):
    # dataloader
    dataset = create_dataset(json_file)
    processor = AutoProcessor.from_pretrained(model_id)

    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    # "microsoft/git-base"
    # model
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # train
    # lr:
    # epoch:
    # batch_size:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()

    for epoch in range(3):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)

            loss = outputs.loss

            # print("Loss:", loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        print("Loss:", loss.item())

    # saving model
    model.save_pretrained(out)


if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
