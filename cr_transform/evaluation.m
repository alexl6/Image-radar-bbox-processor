% This function has two modes to return to different outputs. The first
% mode is to calculate the percent difference between the given parameters,
% and the second mode simply calculates the absolute difference between them. 
% Inputs: calculatedPoints - the user's calculated points
%         truth - the desired theoretical target.
%         mode - 1 to calulate the percent difference, 2 to calculate the
%                absolute difference
% Output: for mode 1 it will output the percent difference the same size as
% the input, for mode 2 it will output the absolute difference between the two and
% will also have the same size as the input.

function diff = evaluation(calculatedPoints, truth, mode)

    if(mode == 1)
       
        diff = 100*(calculatedPoints - truth)./truth;
       
    elseif (mode == 2)
        
        diff = abs(calculatedPoints - truth);
        
    else
        
        error('Please input either 1 or 2 as mode')
        
    end 
        
end