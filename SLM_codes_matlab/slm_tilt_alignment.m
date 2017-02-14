
clear all; %free memory
clc; %clear command window

%% 

%Load image for phasepattern
% PhaseImage=im2double(rgb2gray(imread('Mandela.jpg')));
%PhaseImage=im2double(imread('PupilLogo.bmp'));


Xmax=1920; %Width of the SLM in pixels
Ymax=1080; %Height of the SLM in pixels
PhaseImage = zeros(Ymax, Xmax);

[ydim,xdim]=size(PhaseImage);

PhaseImage=padarray(PhaseImage,[(Ymax-ydim)/2 (Xmax-xdim)/2]);
% PhaseImage=flipdim(PhaseImage,1);
% PhaseImage=flipdim(PhaseImage,2);

X=zeros(Ymax,Xmax); %Allocate memory
Y=zeros(Ymax,Xmax);

%%%send 0th order away with this angle in the x direction
%%% keep it below 0.56 for 2pi/8px
wavelength = 633e-9;
pitch = 8e-6;
theta = 0.1;
theta = deg2rad(theta);
n_px = wavelength / (pitch * tan(theta));
n_px = round(n_px);
actual_angle = rad2deg(atan(wavelength / (pitch * n_px)));
str = ['due to rounding the actual angle will be ', num2str(actual_angle), 'degree'];
disp(str)
str2 = ['and n_px = ', num2str(n_px)];
disp(str2)

tilted = zeros(Ymax, Xmax);
for i = 1:Ymax
    tilted(i, :) = (1)/n_px * mod(i, n_px);
end


for i=1:Ymax
    X(i,:)=(-Xmax/2:Xmax/2-1)'; %Create a matrix with the x-coordinates
end
for i=1:Xmax
    Y(:,i)=-Ymax/2:Ymax/2-1;    %Create a matrix with the y-coordinates
end

Radius=sqrt(X.^2+Y.^2);

mm_radius = 0.2;
max_radius = 0.4;
imradius=mm_radius / max_radius * Ymax/2;
tilted(Radius<imradius)=0;

%imshow(tilted)

colormap('gray')

closescreen()

fullscreen(mat2gray(tilted),2)