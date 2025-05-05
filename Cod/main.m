close all; clear; clc;

%% Incarcare si preprocesare date
T = readtable('diabetes.csv');
X = table2array(T(:,1:end-1));
y = table2array(T(:,end));

[N, n] = size(X);
X = (X - mean(X)) ./ std(X);%normalizare
Xbar = [X, ones(N,1)];

N_train = round(0.8 * N);
N_test = N - N_train;

X_train = Xbar(1:N_train,:);
y_train = y(1:N_train);
X_test = Xbar(N_train+1:end,:);
y_test = y(N_train+1:end);

%% Parametri
m = 15;
max_iter = 10000;
alpha = 1;
batch_size = 32;
tol = 1e-8;

Xw_init = randn(n+1, m)*0.1;
w_init = randn(m, 1)*0.1;

%% Antrenare modele
[Xw_gd, w_gd, loss_gd, norm_grad_gd, time_gd] = GD(X_train, y_train, Xw_init, w_init, alpha, max_iter);
[Xw_sgd, w_sgd, loss_sgd, norm_grad_sgd, time_sgd] = SGD(X_train, y_train, Xw_init, w_init, alpha, max_iter, batch_size);

%% Evaluare
predict_gd = sigmoid(asu(X_test * Xw_gd) * w_gd) >= 0.5;
predict_sgd = sigmoid(asu(X_test * Xw_sgd) * w_sgd) >= 0.5;

fprintf('Matrice confuzie GD:\n');
disp(confusionmat(double(y_test), double(predict_gd)));

fprintf('Matrice confuzie SGD:\n');
disp(confusionmat(double(y_test), double(predict_sgd)));

acc_gd = sum(predict_gd == y_test) / length(y_test);
acc_sgd = sum(predict_sgd == y_test) / length(y_test);

fprintf('Acuratete GD: %.4f\n', acc_gd);
fprintf('Acuratete SGD: %.4f\n', acc_sgd);

%% Rezultate 
figure;
subplot(3,1,1);
semilogx(loss_sgd);
hold on;
semilogx(loss_gd);
legend('SGD','GD');
title('Evolutie Loss');
xlabel('Iteratii');
ylabel('Loss');


subplot(3,1,2);
semilogx(norm_grad_sgd);
hold on;
semilogx(norm_grad_gd);
legend('SGD','GD');
title('Norma Gradientului');
xlabel('Iteratii');
ylabel('Norma Gradientului');


subplot(3,1,3);
semilogx(cumsum(time_sgd));
hold on;
semilogx(cumsum(time_gd));
legend('SGD','GD');
title('Timp Cumulat pe Iteratii');
xlabel('Iteratii');
ylabel('Timp [s]');


C_gd = confusionmat(double(y_test), double(predict_gd));
TN_gd = C_gd(1,1);
FP_gd = C_gd(1,2);
FN_gd = C_gd(2,1);
TP_gd = C_gd(2,2);

precision_gd = TP_gd / (TP_gd + FP_gd);
recall_gd = TP_gd / (TP_gd + FN_gd);
f1_gd = 2 * (precision_gd * recall_gd) / (precision_gd + recall_gd);

fprintf('F1 Score GD: %.4f\n', f1_gd);


C_sgd = confusionmat(double(y_test), double(predict_sgd));
TN_sgd = C_sgd(1,1);
FP_sgd = C_sgd(1,2);
FN_sgd = C_sgd(2,1);
TP_sgd = C_sgd(2,2);

precision_sgd = TP_sgd / (TP_sgd + FP_sgd);
recall_sgd = TP_sgd / (TP_sgd + FN_sgd);
f1_sgd = 2 * (precision_sgd * recall_sgd) / (precision_sgd + recall_sgd);

fprintf('F1 Score SGD: %.4f\n', f1_sgd);
