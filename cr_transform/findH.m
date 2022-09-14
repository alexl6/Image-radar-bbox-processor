% findH is a function that takes in two data sets and then finds the linear
% transformation between the two data sets. For example, if you would like
% to find the mapping between radar and camera, you would input the radar
% as from and camera as to. Format of the data should be row wise(x y z;
% x y z;...]
function H = findH(from, to)
    A0 = [];
    
    for j=1:size(from,1)
        temp =[-from(j,1) -from(j,2) -1 0 0 0 to(j,1)*from(j,1) to(j,1)*from(j,2) to(j,1);
            0 0 0 -from(j,1) -from(j,2) -1 to(j,2)*from(j,1) to(j,2)*from(j,2) to(j,2)];
        A0 = [A0 temp'];
    end
    
    A0 = A0';
    
    % Find h for cam0
    [~,~,V] = svd(A0);
    h0 = V(:,end);

    H = transpose(reshape(h0,3,3));

end