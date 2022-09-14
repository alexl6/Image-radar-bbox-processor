% This function can be used in order to transform data from the initial
% state to the desired on eb y using the previously found H matrix. For
% example, after finding the H between camera and radar, now we wish to
% transform a new set of radar data to camera coordinates, you would
% provide the H matrix relating the two and then the set of radar data.
function calculatedPoints = transform(H, from)

    calculatedPoints = [];
    for j = 1:size(from,1)      
        temp = H*from(j,:)';
        temp = temp./temp(3,:);
        calculatedPoints = [calculatedPoints temp];
    end


end