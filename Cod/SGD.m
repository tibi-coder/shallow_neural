function [Xw, w, loss, norm_grad, time_vec] = SGD(X, y, Xw_init, w_init, alpha, max_iter, batch_size)
    Xw = Xw_init;
    w = w_init;
    loss = zeros(max_iter, 1);
    norm_grad = zeros(max_iter, 1);
    time_vec = zeros(max_iter, 1);
    
    tol = 1e-8; % toleranta pe norma gradientului
    
    for iter = 1:max_iter
        tic;
        
        % Batch random
        idx_batch = randsample(length(y), batch_size);
        X_batch = X(idx_batch,:);
        y_batch = y(idx_batch);
        
        % Forward
        Z = X_batch * Xw;
        G = asu(Z);
        y_pred = sigmoid(G*w);

        % Loss
        loss(iter) = loss_function(y_batch, y_pred);

        % Backward
        delta = (y_pred - y_batch) .* y_pred .* (1 - y_pred);
        dG = asu_deriv(Z);

        grad_w = G' * delta / batch_size;
        grad_Xw = (X_batch' * (delta * w' .* dG)) / batch_size;

        % Norm gradient
        norm_grad(iter) = norm([grad_Xw(:); grad_w]);
        
        % Conditie de oprire
        if norm_grad(iter) < tol
            loss = loss(1:iter);
            norm_grad = norm_grad(1:iter);
            time_vec = time_vec(1:iter);
            break;
        end

        % Update parametri
        w = w - alpha * grad_w;
        Xw = Xw - alpha * grad_Xw;
        
        time_vec(iter) = toc;
    end
end
