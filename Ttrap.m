function [ FreeSurf ] = Ttrap( Bottom,Sideslope,Head )
%PTRAP calculates the free surface of the atrapezoidal channel
%   Inputs: Bottom width, slope of the inclined side walls and water head
%           inside channel.

FreeSurf=Bottom+2*Head*Sideslope;

end