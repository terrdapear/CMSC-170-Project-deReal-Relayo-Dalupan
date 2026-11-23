import pickle

def save_model(model, name, fol):
    try:
        with open(f"{fol}/{name}.pkl","wb") as f:
            pickle.dump(model,f)
        return True
    except Exception:
        return False

def load_model(path):
    try:
        with open(f"{path}","rb") as f:
            return pickle.load(f)
    except Exception:
        return None

