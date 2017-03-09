function [ img ] = snapImage(g, exposuretime)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    %% acquision of image intensity
    % create video object
    g.ExposureTime = exposuretime;
    img = snapshot(g);
end

