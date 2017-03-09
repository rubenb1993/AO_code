
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
theta = 0.5;
theta = deg2rad(theta);
n_px = wavelength / (pitch * tan(theta));
n_px = round(n_px);
actual_angle = rad2deg(atan(wavelength / (pitch * n_px)));
str = ['due to rounding the actual angle will be ', num2str(actual_angle), 'degree'];
disp(str)
str2 = ['and n_px = ', num2str(n_px)];
disp(str2)

%make tilt matrix
tilted = zeros(Ymax, Xmax);
for i = 1:Xmax
    tilted(:, i) = (1)/(n_px-1) * mod(i, (n_px));
    %tilted(:, i) = 1/n_px * i;
end
for i=1:Ymax
    X(i,:)=(-Xmax/2:Xmax/2-1)'; %Create a matrix with the x-coordinates
end
for i=1:Xmax
    Y(:,i)=-Ymax/2:Ymax/2-1;    %Create a matrix with the y-coordinates
end

%define aperture
Radius=sqrt(X.^2+Y.^2);
mm_radius = 0.18;
max_radius = 0.4;
radius_mod = zeros(Ymax, Xmax);
imradius=mm_radius / max_radius * Ymax/2;
radius_mod(Radius<imradius) = 1;
%tilted(Radius<imradius)=0.0;
circular = tilted;
circular(Radius>imradius)=0.0;

% make L
L = zeros(Ymax, Xmax);
L(Ymax/2 - 80:Ymax/2, Xmax/2-100:Xmax/2+100) = 1;
L(Ymax/2-350:Ymax/2-50, Xmax/2-100:Xmax/2) = 1;
first_order_L = mod(L .* tilted, 1);


%make cross
cross= zeros(Ymax, Xmax);
dev = 10;
cross(Ymax/2-dev:Ymax/2+dev, :) = 1;
cross(:, Xmax/2-dev:Xmax/2+dev) = 1;
first_order_cross = mod(cross .* tilted, 1);



% make defocus
Radius_norm = Radius / (Ymax/2);

% 
% subplot(3,1,1), imshow(L);
% subplot(3,1,2), imshow(tilted);
% subplot(3,1,3), imshow(first_order_L);
% colormap('gray')
% figure
% subplot(3,1,1), imshow(cross);
% subplot(3,1,2), imshow(tilted);
% subplot(3,1,3), imshow(first_order_cross);
% figure
% imshow(first_order_def);
% figure
% plot(first_order_def(Ymax/2, Xmax/2 - Ymax/2 : Xmax/2 + Ymax/2)); %
i = 1
while 1
    a_def = sin(i/20*2*pi) * n_px;
    defocus = a_def * (2 * Radius_norm.^2 - 1); 
    first_order_def = radius_mod .* (1/n_px * mod(defocus + tilted, n_px));
    pause(2)
    closescreen()
% 
    fullscreen(mat2gray(first_order_def),2)
    i = i + 1;
end
