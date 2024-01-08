function [loss,gradients,state] = modelLoss(net,X,T)
    [Y,state] = forward(net,X);
    loss1 = 0.25*crossentropy(Y,T);
    loss2 = mse(Y,T);

    fmap = forward(net,X,"Outputs","relu_3");
    fvec = reshape(fmap,[],size(fmap,4));
    [~, labels] = max(T);
    all_labels = 1:10;
    c = single(zeros(size(fvec,1),length(all_labels)));
    s = single(zeros(size(fvec,1),length(all_labels)));
    for k = all_labels
        c(:,k) = mean(fvec(:,labels == k),2);
        s(:,k) = std(fvec(:,labels == k),[],2);
    end

    center_dist = single(zeros(length(all_labels)));
    for k1 = all_labels
        for k2 = all_labels
            center_dist(k1,k2) = norm(c(:,k1) - c(:,k2));
        end
    end

    loss3 = 100*(1/mean(center_dist,'all')^2);
    loss4 = 10*(std(mean(s)))^2;
    loss5 = 0.005*(sum(mean(s)))^2;
    
    loss = loss1 + loss2 + loss3 + loss4 + loss5;
    
    figure(1); 
    bar(double(string([loss1,loss2,loss3,loss4,loss5])));
    title("loss variation per iteration");

    gradients = dlgradient(loss,net.Learnables);
end

