% This code solves steady-state 2D conduction equation
% assumptions: dx=dy
%              domain is square

% Important Note: Scheme is cell vertex.
clear
clc
%% Settings

nxmax=101; % Number of vertices in x-direction
nymax=101; % Number of vertices in y-direction
StdError=0.01;
Lx=2;
Ly=Lx;
dx=Lx/(nxmax-1);
dy=dx;

%% Mesh Generation
x(1,1:nymax)=0;
for i=2:nxmax
   for j=1:nymax
        x(i,j)=x(i-1,j)+dx;
    end
end

y(1:nymax,1)=0;
for i=1:nxmax
   for j=2:nymax
        y(i,j)=y(i,j-1)+dy;
    end
end

%% Initial Condition
Tnew(1:nxmax,1:nymax)=30;
Told(1:nxmax,1:nymax)=30;

%% Boundary Condition
% Left Boundary
Told(1,:)=20;
Tnew(1,:)=20;
% Right Boundary
Told(nxmax,:)=20;
Tnew(nxmax,:)=20;
% Bottom Boundary
Told(:,1)=40;
Tnew(:,1)=40;
% Top Boundary
Told(:,nymax)=40;
Tnew(:,nymax)=40;

%% Processing

Tdiff=StdError+StdError;
while norm(Tdiff) > StdError

for i=2:nxmax-1
   for j=2:nymax-1 
    
      Tnew(i,j)=0.25*(Told(i+1,j)+Told(i,j+1)+Told(i-1,j)+Told(i,j-1));
       
   end
end

Tdiff=0;
for i=1:nxmax
   for j=1:nymax
       Tdiff=abs(Tnew(i,j)-Told(i,j))+Tdiff;
   end 
end

disp([ ' Error = ' num2str(Tdiff)])

Told=Tnew;
end

%% Post-Processing
contourf(x,y,Tnew)

