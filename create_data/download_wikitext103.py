from torchtext.datasets import WikiText103

if __name__ == "__main__":
    data_dir = '../data'
    WikiText103.download(root=data_dir)
