function [ img ] = complete_image( exposuretime  )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    g = gigecam;
    g.Height = 1024;
    g.Width = 1024;
    g.ExposureTime = exposuretime;
    img = snapshot(g);
    clear g
end

