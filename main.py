from utils.image_analysis import display_image, display_json, display_annotation_on_image


if __name__ == "__main__":
    # img_path = r"./dataset/training_data/images/0000971160.png"
    # img = display_image(img_path, 'return')

    # json_path = r'./dataset/training_data/annotations/0000971160.json'
    # json_data = display_json(json_path, 'return')

    # display_annotation_on_image(img, json_data)

    from transformers import LayoutLMModel, LayoutLMTokenizer

    model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")