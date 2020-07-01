function [ Area ] = Atrap( Bottom,Sideslope,Head )
%ATRAP calculates the wetted area of the atrapezoidal channel
%   Inputs: Bottom width, slope of the inclined side walls and water head
%           inside channel.
%   Output: Wetted Area

Area=Head*(Bottom+Head*Sideslope);

end