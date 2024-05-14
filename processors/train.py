from tqdm import tqdm

from transformers import AdamW
from transformers import LayoutLMForTokenClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "NUM_TRAIN_EPOCHS": 2,
    "Learning_RATE": 5e-5
}

class train():
    def __init__(self, num_labels):
        self.model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels)
        self.model.to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=config["Learning_RATE"])


    def start_training(self, train_dataloader):
        self.model.train()
        global_step = 0
        # t_total = len(train_dataloader) * config["NUM_TRAIN_EPOCHS"] # total number of training steps
        for epoch in range(config["NUM_TRAIN_EPOCHS"]):
            for batch in tqdm(train_dataloader, desc="Training"):
                input_ids = batch[0].to(device)
                bbox = batch[4].to(device)
                attention_mask = batch[1].to(device)
                token_type_ids = batch[2].to(device)
                labels = batch[3].to(device)

                # forward pass
                outputs = self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=labels)
                loss = outputs.loss

                # print loss every 100 steps
                if global_step % 100 == 0:
                    print(f"Loss after {global_step} steps: {loss.item()}")

                # backward pass to get the gradients 
                loss.backward()

                #print("Gradients on classification head:")
                #print(model.classifier.weight.grad[6,:].sum())

                # update
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1


# http://10.128.68.171/