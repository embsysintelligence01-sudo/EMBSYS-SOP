"""Original engines/purity/identity_features.py definitions.

Loaded into tracker globals by the explicit temporary compatibility binder.
The source bodies are unchanged; see docs/IMPORT_PATH_MIGRATION.md.
"""

class JewelIdModel:
    """Per-session jewel identities (16-10): each jewel captured when the stone was placed is a class, plus a
    "not a jewel" class from bench and stone crops (rub marks). ImageNet ResNet18 (frozen) turns a crop into 512
    features; a linear softmax classifier on them is trained from ID_SAMPLES augmented copies per jewel."""

    def __init__(self):
        self.session = self.net = None
        onnx_path = str(model_path("shared.identity_features"))
        if os.path.exists(onnx_path):
            self.session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        else:  # torch fallback
            import torchvision
            net = torchvision.models.resnet18(weights=None)
            path = str(model_path("shared.identity_checkpoint"))
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            net.fc = torch.nn.Identity()
            self.net = net.eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self.labels, self.W, self.b = [], None, None
        self.ready = False

    def features(self, crops, batch=32):
        out = []
        for k in range(0, len(crops), batch):
            xs = []
            for c in crops[k:k + batch]:
                h, w = c.shape[:2]
                side = max(h, w)
                sq = cv2.copyMakeBorder(c, (side - h) // 2, side - h - (side - h) // 2, (side - w) // 2,
                                        side - w - (side - w) // 2, cv2.BORDER_REPLICATE)
                x = cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
                xs.append(((x - self.mean) / self.std).transpose(2, 0, 1))
            if self.session is not None:
                f = self.session.run(None, {self.session.get_inputs()[0].name: np.stack(xs).astype(np.float32)})[0]
            else:
                with torch.inference_mode():
                    f = self.net(torch.from_numpy(np.stack(xs))).numpy()
            out.append(f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-6))
        return np.concatenate(out)

    def train(self, crops_by_id, negatives, seed=0):
        """crops_by_id: {jewel id: capture crop}; negatives: bench / stone crops. Returns the training accuracy
        on fresh augmented copies (a self-check)."""
        rng = np.random.default_rng(seed)
        labels = sorted(crops_by_id) + [None]
        xs, ys = [], []
        for k, jid in enumerate(labels[:-1]):
            c = crops_by_id[jid]
            xs += [c] + [augment(c, rng) for _ in range(ID_SAMPLES)]
            ys += [k] * (ID_SAMPLES + 1)
        for c in negatives:
            xs.append(c)
            ys.append(len(labels) - 1)
        X = torch.from_numpy(self.features(xs))
        y = torch.tensor(ys)
        lin = torch.nn.Linear(X.shape[1], len(labels))
        opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-4)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lin(X) * 10.0, y)  # x10: features are unit length
            loss.backward()
            opt.step()
        self.labels = labels
        self.W, self.b = lin.weight.detach().numpy() * 10.0, lin.bias.detach().numpy() * 10.0
        check = [augment(crops_by_id[j], rng) for j in labels[:-1] for _ in range(6)]
        truth = [j for j in labels[:-1] for _ in range(6)]
        pred = [self.predict_features(f)[0] for f in self.features(check)]
        self.ready = True
        return float(np.mean([a == b for a, b in zip(pred, truth)]))

    def predict_features(self, f):
        z = self.W @ f + self.b
        p = np.exp(z - z.max())
        p /= p.sum()
        k = int(np.argmax(p))
        return self.labels[k], float(p[k])

    def predict(self, crop):
        """(jewel id or None for "not a jewel", probability)"""
        if not self.ready or crop is None or crop.size == 0:
            return None, 0.0
        return self.predict_features(self.features([crop])[0])

