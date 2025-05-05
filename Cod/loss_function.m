function L = loss_function(e, y_pred)
    L = -(1/length(e)) * sum(e.*log(y_pred + 1e-8) + (1-e).*log(1 - y_pred + 1e-8));
end
