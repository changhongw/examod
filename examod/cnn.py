import os, pandas as pd, numpy as np, torch, gin
from torch import nn
from torchmetrics import Accuracy, ClasswiseWrapper
from pytorch_lightning.core.lightning import LightningModule
from examod.utils import load_as_tensor

gin.enter_interactive_mode()

@gin.configurable
class SOLdatasetClassifier(LightningModule):
    def __init__(self,
                 c = 1e-1,
                 lr=1e-3,
                 average='weighted',
                 stats_dir='SOL-0.9HQ-PMT/',
                 csv='../datasets/SOL-0.9HQ-PMT/SOL-0.9HQ-PMT_meta.csv',
                 feature='scat1d_s1s2',
                 n_batches_train = None):
        super().__init__()

        self.lr = lr
        self.feature = feature.split('_')[0]
        self.feature_spec = feature.split('_')[1]
        self.n_batches_train = n_batches_train

        df = pd.read_csv(csv)
        classes = df[['modulation technique', 'label']
                    ].value_counts().index.to_list()
        classes = [x[0] for x in sorted(classes, key=lambda x: x[1])]
        
        supports_unsorted = df['modulation technique'].value_counts(sort=False)
        supports = [supports_unsorted[x] for x in classes]
        class_weight = [max(supports) / s for s in supports]
        class_weight = torch.tensor(class_weight, dtype=torch.float32)

        self.acc_metric = Accuracy(num_classes=len(classes), average=average)
        self.acc_metric_macro = Accuracy(num_classes=len(classes), average='macro')

        self.classwise_acc = ClasswiseWrapper(
            Accuracy(num_classes=len(classes), average=None), labels=classes)
        self.loss = nn.CrossEntropyLoss(weight=class_weight)

        self.val_acc = None
        self.val_loss = None

        stats_dir = os.path.join(os.getcwd(), stats_dir, self.feature)
        self.c = c

        if self.feature_spec=='s1s2':
            self.mu = load_as_tensor(os.path.join(stats_dir, 'stats','mu_S1S2.npy'))
            self.n_channels = len(self.mu) 
        else:
            self.mu = load_as_tensor(os.path.join(stats_dir, 'stats','mu_S1.npy'))
            self.n_channels = len(self.mu) 
            
        self.bn = (nn.BatchNorm1d(self.n_channels))

        self.setup_cnn(len(classes))  # initialize CNN

        self.automatic_optimization = False

    def setup_cnn(self, num_classes):
        self.conv_net = CNN1D(self.n_channels, num_classes=num_classes) 

    def forward(self, x):
        x = self.bn(x)  
        # convnet
        y = self.conv_net(x)

        return y

    def step(self, batch, fold):
        Sx, y = batch    
        logits = self(Sx)

        loss = self.loss(logits, y)

        return {'loss': loss,
                'logits': logits,
                'y': y}

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        def closure():
            opt.zero_grad()
            self.manual_backward(loss)
            return loss

        self.update_lr(batch_idx)
        opt.step(closure=closure)

        return {'loss': loss,
                'logits': logits,
                'y': y}

    def validation_step(self, batch, batch_idx):
        return self.step(batch, fold='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, fold='test')

    def training_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs]).softmax(dim=-1)
        y = torch.cat([x['y'] for x in outputs])
        acc_macro = self.acc_metric_macro(logits, y)
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('train/acc', acc_macro)
        self.log('train/loss', loss, prog_bar=True)

        self.reset_metrics()

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs]).softmax(dim=-1)
        y = torch.cat([x['y'] for x in outputs])
        acc_macro = self.acc_metric_macro(logits, y)

        self.val_acc = acc_macro
        self.val_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('val/acc', self.val_acc, on_step=False,
                 prog_bar=True, on_epoch=True)
        self.log('val/loss', self.val_loss, on_step=False,
                 prog_bar=True, on_epoch=True)

        self.reset_metrics()

    def test_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs]).softmax(dim=-1)
        y = torch.cat([x['y'] for x in outputs])
        acc_macro = self.acc_metric_macro(logits, y)
        
        np.save('results/testresluts_truth_pred_' + self.feature_spec + '.npy', np.vstack((y, np.argmax(logits, -1))))

        bin_counts = torch.bincount(y)
        classwise_acc = [torch.zeros(n) for n in bin_counts]
        class_counts = [0 for _ in bin_counts]
        preds = logits.argmax(dim=-1)
        for i, p in enumerate(preds):
            score = float(preds[i] == y[i])
            classwise_acc[y[i]][class_counts[y[i]]] = score
            class_counts[y[i]] += 1

        acc_classwise = {i: float(acc.mean())
                         for i, acc in enumerate(classwise_acc)}

        if self.val_acc is not None:
            self.log(f'val_acc', self.val_acc)
            self.log(f'val_loss', self.val_loss)
        self.log('acc_macro', acc_macro)
        self.log('acc_classwise', acc_classwise)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=1e-1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=1, eta_min=1e-8,
            last_epoch=-1, verbose=0)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                },
            }

    def update_lr(self, batch_idx):
        sch = self.lr_schedulers()

        warmup_epochs = 3
        warmup_len = self.n_batches_train * warmup_epochs
        total_step = self.trainer.current_epoch * self.n_batches_train + batch_idx
        if total_step >= warmup_len:
            epoch_frac = total_step / self.n_batches_train
        else:
            # LR warmup for first epoch
            # `batch_idx + 1` to not start with `1` when `batch_idx == 0`
            epoch_frac = 1 - (total_step + 1) / warmup_len
        sch.step(epoch_frac)
        return sch

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=3,
            gradient_clip_algorithm='norm',
        )

    def reset_metrics(self):
        self.acc_metric_macro.reset()
        self.classwise_acc.reset()

class CNN1D(nn.Module):
    def __init__(self, in_channels, num_classes, dense_dim=64, drop_rate=.5):
        super().__init__()
        c_ref = in_channels // 4
        ckw = dict(stride=1, bias=False, padding='same')
        C0, C1, C2 = c_ref//2, c_ref, 2*c_ref

        self.conv0 = nn.Conv1d(in_channels=in_channels, out_channels=C0,
                               kernel_size=7, **ckw)
        self.bn0 = nn.BatchNorm1d(C0)
        self.ap0 = nn.AvgPool1d(kernel_size=2)

        self.conv1 = nn.Conv1d(in_channels=C0, out_channels=C1,
                               kernel_size=5, **ckw)
        self.bn1 = nn.BatchNorm1d(C1)
        self.ap1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=C1, out_channels=C2,
                               kernel_size=3, **ckw)
        self.bn2 = nn.BatchNorm1d(C2)

        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc = nn.Linear(in_features=C2, out_features=dense_dim)
        self.dp = nn.Dropout(drop_rate)
        self.fc_out = nn.Linear(in_features=dense_dim, out_features=num_classes)

    def forward_features(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.ap0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.ap1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # global average over time axis
        x = self.avgpool(x)
        x = x.squeeze()
        
        return x

    def classifier(self, x):
        x = self.dp(x)

        x = self.fc(x)
        x = self.relu(x)

        x = self.fc_out(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x