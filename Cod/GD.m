function [Xw, w, loss, norm_grad, time_vec] = GD(X, y, Xw_init, w_init, alpha, max_iter)
    Xw = Xw_init;
    w = w_init;
    loss = zeros(max_iter, 1);
    norm_grad = zeros(max_iter, 1);
    time_vec = zeros(max_iter, 1);
    
    tol = 1e-8; % toleranta pe norma gradientului
    
    for iter = 1:max_iter
        tic;
        
        % Forward
        Z = X * Xw;
        G = asu(Z);
        y_pred = sigmoid(G*w);

        % Loss
        loss(iter) = loss_function(y, y_pred);

        % Backward
        delta = (y_pred - y) .* y_pred .* (1 - y_pred);
        dG = asu_deriv(Z);

        grad_w = G' * delta / length(y);
        grad_Xw = (X' * (delta * w' .* dG)) / length(y);

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
