
clear all; %free memory
clc; %clear command window

%% 

%Load image for phasepattern
% PhaseImage=im2double(rgb2gray(imread('Mandela.jpg')));
PhaseImage=im2double(imread('PupilLogo.bmp'));

Xmax=1920; %Width of the SLM in pixels
Ymax=1080; %Height of the SLM in pixels

[ydim,xdim]=size(PhaseImage);

PhaseImage=padarray(PhaseImage,[(Ymax-ydim)/2 (Xmax-xdim)/2]);
% PhaseImage=flipdim(PhaseImage,1);
% PhaseImage=flipdim(PhaseImage,2);

X=zeros(Ymax,Xmax); %Allocate memory
Y=zeros(Ymax,Xmax);


for i=1:Ymax
    X(i,:)=(-Xmax/2:Xmax/2-1)'; %Create a matrix with the x-coordinates
end
for i=1:Xmax
    Y(:,i)=-Ymax/2:Ymax/2-1;    %Create a matrix with the y-coordinates
end

Radius=sqrt(X.^2+Y.^2);

imradius=250;
PhaseImage(Radius>imradius)=0;

buffer_tiltx=0;
buffer_tilty=-500;

buffer_Tilt=mod(buffer_tiltx*X/Xmax+buffer_tilty*Y/Ymax,1);

PhaseImage=(Radius>imradius).*buffer_Tilt+PhaseImage;

% PhaseImage=(Radius>imradius).*buffer_Tilt;

%% Introduce phase shift for reconstruction

A_radius=10; %Pixel size
A=0/3; %Shift phase of DC component

PhaseImage(Radius<=A_radius)=mod(PhaseImage(Radius<=A_radius)+A,1);

%% Create the aberrations

F=4.0;

Defocus=F*(Radius/imradius).^2;


% %% Create tilt

tiltx=600;
tilty=-50;

Tilt=mod(tiltx*X/Xmax+tilty*Y/Ymax,1);

% %% Combine the images
ShiftX=0;
ShiftY=0;

Defocus=circshift(Defocus,[-ShiftY,-ShiftX]);
Defocus=Defocus.*(Radius<=imradius);
PhaseImage=circshift(PhaseImage,[-ShiftY,-ShiftX]);

CombinedImage=mod(Defocus+Tilt+PhaseImage,1);

% %% Assign image to second screen (=SLM)

% imagesc(PhaseImage)
% colormap('gray')

closescreen()

fullscreen(mat2gray(CombinedImage),2)