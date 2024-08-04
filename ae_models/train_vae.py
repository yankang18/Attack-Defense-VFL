import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

from ae_models.vae_models import VAE
from show_autoencoder import show_autoencoder_transform_result
from utils import cross_entropy_for_one_hot
from utils import get_timestamp, sharpen, entropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def kl_divergence_loss(q_dist):
    return kl_divergence(
        q_dist, Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.stddev))
    ).sum(-1)


def label_to_one_hot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    one_hot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    one_hot_target.scatter_(1, target, 1)
    return one_hot_target


def train_batch(model, optimizer, code_y, hyperparameter_dict):
    label_y = torch.argmax(code_y, dim=1)
    # OH_y = label_to_one_hot(label_y, num_classes=dim)

    # lambda_1 = hyperparameter_dict["lambda_1"]
    entropy_lbda = hyperparameter_dict["entropy_lbda"]

    # print("-"*100)
    # print("code_y", code_y)
    # =================== forward =====================
    decoding, q_dist, z_sample = model(code_y)

    label_y_hat = torch.argmax(decoding, dim=1)
    label_y_enc = torch.argmax(z_sample, dim=1)
    # print("code_D_y", code_D_y)
    loss_e = entropy(z_sample)
    loss_p = cross_entropy_for_one_hot(decoding, code_y, reduce="sum")
    # loss_p = criterion(code_y_hat, label_y)
    loss_n = cross_entropy_for_one_hot(z_sample, code_y, reduce="sum")
    # loss = 10 * loss_p - lambda_2 * loss_e - loss_n
    # loss = 10 * loss_p - lambda_2 * loss_e
    loss_kl = kl_divergence_loss(q_dist).sum()
    loss = 10 * loss_p + 1 * loss_kl - entropy_lbda * loss_e - loss_n

    # =================== backward ====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc_p = torch.sum(torch.eq(label_y_hat, label_y)) / float(len(label_y))
    train_acc_n = torch.sum(torch.ne(label_y_enc, label_y)) / float(len(label_y))
    # train_acc_n = torch.sum(label_y_enc != label_y) / float(len(label_y))
    train_loss = loss.item()

    # =================== log =========================
    # print(f"loss_p:{loss_p.item()}, loss_e:{loss_e.item()}, loss_n:{loss_n.item()}")
    # loss_dict = {"loss_p": loss_p.item(), "loss_e": loss_e.item(), "loss_n": loss_n.item()}
    # loss_dict = {"loss_p": loss_p.item(), "loss_e": loss_e.item(), "loss_n": 0}
    loss_dict = {"loss_p": loss_p.item(), "loss_e": loss_e, "loss_n": loss_n.item(), "loss_kl": loss_kl}
    return train_loss, train_acc_p, train_acc_n, loss_dict


if __name__ == '__main__':
    # Training and testing
    # num_classes = 100
    # num_classes = 20
    num_classes = 10
    # num_classes = 5
    # num_classes = 2

    vae_model = VAE(z_dim=num_classes, input_dim=num_classes, hidden_dim=(2 + (num_classes * 6)) ** 2).to(device)
    # vae_model = VAE(z_dim=num_classes, input_dim=num_classes, hidden_dim=(num_classes * 2) ** 2).to(device)
    # learning_rate = 1e-4
    learning_rate = 5e-4
    batch_size = 128
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

    hyperparameter_dict = dict()
    # hyperparameter_dict["lambda_1"] = 10
    # TODO: 0.1, 0.5, 1.0, 1.5, 2.0
    entropy_lbda = 0.01
    hyperparameter_dict["entropy_lbda"] = entropy_lbda
    print("[INFO] entropy_lbda:{}".format(entropy_lbda))

    # epochs = 50
    # T = 0.05
    # epochs = 100  # train 20 classes
    # T = 0.028
    epochs = 100  # train 10 classes
    T = 0.025
    # epochs = 20  # train 5 classes
    # T = 0.025
    # epochs = 50  # train 2 classes
    # T = 0.025

    # criterion = torch.nn.TripletMarginLoss(margin=2.0)
    criterion = torch.nn.CrossEntropyLoss()
    train_sample_size = 30000
    rand_train_x = torch.rand(train_sample_size, num_classes)
    train_y = sharpen(F.softmax(rand_train_x, dim=1), T=T)
    # train_y = F.softmax(rand_train_x, dim=1)

    print(train_y[0])

    test_sample_size = 10000
    rand_test_x = torch.rand(test_sample_size, num_classes)
    test_y = sharpen(F.softmax(rand_test_x, dim=1), T=T)
    # test_y = F.softmax(rand_test_x, dim=1)

    print(f"train data: {train_y[0]}")
    for epoch in range(0, epochs + 1):
        iteration = 0
        train_loss = 0
        code_y = train_y.to(device)
        for batch_start_idx in range(0, code_y.shape[0], batch_size):
            iteration += 1
            # print(batch_start_idx, batch_start_idx + batch_size)
            batch_code_y = code_y[batch_start_idx:batch_start_idx + batch_size]
            train_loss, train_acc_p, train_acc_n, loss_dict = train_batch(vae_model,
                                                                          optimizer,
                                                                          batch_code_y,
                                                                          hyperparameter_dict)

            if (iteration + 1) % 50 == 0:
                # validation on test data
                test_code_y = test_y.to(device)
                test_label_y = torch.argmax(test_code_y, dim=1)
                test_code_y_hat, _, test_code_D_y = vae_model(test_code_y)
                # print("test_code_D_y: \n", test_code_D_y[0])
                test_label_y_hat = torch.argmax(test_code_y_hat, dim=1)
                test_label_D_y = torch.argmax(test_code_D_y, dim=1)

                # test_acc_p = torch.sum(tst_label_y_hat == test_label_y) / float(len(test_label_y))
                # test_acc_n = torch.sum(tst_label_D_y != test_label_y) / float(len(test_label_y))
                test_acc_p = torch.sum(torch.eq(test_label_y_hat, test_label_y)) / float(len(test_label_y))
                test_acc_n = torch.sum(torch.ne(test_label_D_y, test_label_y)) / float(len(test_label_y))

                print("-" * 100)
                print(f"[INFO] epoch:{epoch}, iter:{iteration}, loss:{train_loss}")
                print(f"[INFO] loss_p:{loss_dict['loss_p']}, loss_kl:{loss_dict['loss_kl']}, loss_n:{loss_dict['loss_n']}")
                print(f"[INFO]   train acc p : {train_acc_p}, train acc n : {train_acc_n}")
                print(f"[INFO]   test acc p : {test_acc_p}, test acc n : {test_acc_n}")

    show_autoencoder_transform_result(vae_model, num_classes)

    timestamp = get_timestamp()
    model_name = f"vae_{num_classes}_{entropy_lbda}_{timestamp}"
    model_full_path = f"./vae_trained_models/{model_name}"
    vae_model.save_model(model_full_path)
    print(f"[INFO] save model to:{model_full_path}")
