from .noseawater_loader import NotSeawaterSegmentation

datasets = {
    'notseawater': NotSeawaterSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
