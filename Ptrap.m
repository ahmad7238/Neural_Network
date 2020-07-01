function [ Perimeter ] = Ptrap( Bottom,Sideslope,Head )
%PTRAP calculates the wetted perimeter of the atrapezoidal channel
%   Inputs: Bottom width, slope of the inclined side walls and water head
%           inside channel.

Perimeter=Bottom+2*Head*sqrt(1+Sideslope^2);

end