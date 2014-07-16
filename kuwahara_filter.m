function [ I_f ] = kuwahara_filter( I, L )
% KUWAHARA_FILTER   Traditional Kuwahara filter implementation.
%   [I_f] = KUWAHARA_FILTER(I,L) takes a RGB image (uint8) I with a kernel
%   bandwidth modifier L (default = 1) and applies the Kuwahara filter on
%   it, producing the filtered image I_f.
%
%   Parameters:
%   I: RGB image (uint8)
%   L: kernel bandwidth param, kernel size = 4*L+1
%   I_f: filtered image (RGB, uint8) 
%
%	Author: Carlos Wang
%	Created: August 28, 2010
%   Updated: January 10, 2011
%
%   Resources used:
%   http://www.cse.ust.hk/learning_objects/imageprocessing/kuwahara/kuwahara.html
%   http://www.ph.tn.tudelft.nl/Courses/FIP/noframes/fip-Smoothin.html#Heading88

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%assign defaults/simple checks
I_f = [];
if nargin < 1   %no arguments
    disp('Error: KUWAHARA_FILTER(I,L) has no input arguments.');
    return;
end
if nargin < 2   %default L value
    L = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize
P = 4*L + 1;            %kernel size
P_R = 2*L + 1;          %region size
P_R1 = P_R-1;           %region size without overlap

I_p = padarray(I, [P_R1 P_R1], 'replicate'); %pad image to account for image boundary
I_d = im2double(I_p);         %convert image to double
I_hsv = rgb2hsv(I_p);         %find hsv equivalent
I_f = zeros(1,1,3);           %initialize filtered image
m = length(I_p(:,1,1));       %image size (rows)
n = length(I_p(1,:,1));       %image size (columns)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%find mean colour (rgb) and variance (value channel) for each region
Rm = zeros(m,n,3);  %region colour mean placeholder
Rv1 = zeros(1,1);   %value variance 1 placeholder
Rv2 = zeros(1,1);   %value variance 2 placeholder
Rv = zeros(1,1);    %region value variance placeholder
A_R = P_R*P_R;      %area of region
kernel_m = ones(P_R)/A_R;   %mean kernel
kernel_v = ones(P_R);       %variance kernel (used to find variance)
I_v = I_hsv(:,:,3);         %v channel
for k=1:3
    Rm(:,:,k) = filter2(kernel_m, I_d(:,:,k));  %calculate mean colour per region
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%find v channel variance per region
Rv1 = filter2(kernel_v, I_v.^2)/(A_R-1);            %first term
Rv2 = filter2(kernel_v, I_v).^2/((A_R*A_R)-A_R);    %second term
Rv = Rv1-Rv2;                                       %variance per region

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%apply Kuwahara filter over image
padRv_v = padarray(Rv, [P_R1, 0], -inf);
padRv_h = padarray(Rv, [0, P_R1], -inf);
padRv_a = padarray(Rv, [P_R1, P_R1], -inf);
lpadRv = size(padRv_a,2);      %length
hpadRv = size(padRv_a,1);      %height

%shift Rv on all directions (ESWN/corners). this is used to compare values
%from all regions to their neighbouring regions.
upvec = [P_R1*2+1:hpadRv]; downvec = [1:hpadRv-P_R1*2]; leftvec = [P_R1*2+1:lpadRv]; rightvec = [1:lpadRv-P_R1*2];
padRv_up = padRv_v(upvec,:); padRv_down = padRv_v(downvec,:);
padRv_left = padRv_h(:,leftvec); padRv_right = padRv_h(:,rightvec);
padRv_upr = padRv_a(upvec,rightvec); padRv_upl = padRv_a(upvec,leftvec);
padRv_downl = padRv_a(downvec,leftvec); padRv_downr = padRv_a(downvec,rightvec);

%compare each region variance (matrix way of comparing)
c_upl = (padRv_up>=Rv & padRv_upl>=Rv & padRv_left>=Rv).*1;          %top left is smallest
c_upr = (padRv_up>=Rv & padRv_upr>=Rv & padRv_right>=Rv).*2;         %top right is smallest
c_downl = (padRv_down>=Rv & padRv_downl>=Rv & padRv_left>=Rv).*4;    %bot left is smallest
c_downr = (padRv_down>=Rv & padRv_downr>=Rv & padRv_right>=Rv).*8;   %bot right is smallest

%shift and add results
c_upr = [c_upr(:,P_R1+1:end) zeros(length(c_upr(:,1)),P_R1)];
c_downl = [c_downl(P_R1+1:end,:); zeros(P_R1,length(c_downl(1,:)))];
c_downr = padarray(c_downr, [P_R1, P_R1]);
c_downr = c_downr(P_R1*2+1:end, P_R1*2+1:end);
c_result = c_upl + c_upr + c_downl + c_downr;
c_result = (c_result == 1) ...                                  %use top left region
                + ((c_result >= 2) & (c_result <= 3)).*2 ...    %use top right region
                + ((c_result >= 4) & (c_result <= 7)).*3 ...    %use bot left region
                + ((c_result >= 8) & (c_result <= 16)).*4;      %use bot left region

%create indexing matrix
sizeRv = numel(Rv);
idxM = 1:sizeRv;                %vector index
idxM = reshape(idxM,size(Rv));  %put in matrix form
idxM = repmat(idxM, [1 1 4]);   %replicate onto different levels (4 levels for the four regions)
idxM(:,:,2) = [idxM(:,P_R1+1:end,2) zeros(size(idxM,1),P_R1)];      %define regions (upper right = 2)
idxM(:,:,3) = [idxM(P_R1+1:end,:,3) ; zeros(P_R1,size(idxM,2))];    %define regions (bot left = 3)
tmpM = padarray(idxM(:,:,4), [P_R1, P_R1]);
idxM(:,:,4) = tmpM(P_R1*2+1:end, P_R1*2+1:end);                     %define regions (bot right = 4)
c_result_idx = (c_result-1).*((c_result-1)>=0).*sizeRv+idxM(:,:,1); %find from idxM the location of the region (vector index)
c_result_idx = idxM(c_result_idx(:));                               %index vectors of results

%set results (rgb values) to the filtered image
tmpRm = Rm(:,:,1);
I_f = reshape(tmpRm(c_result_idx), size(Rv));
tmpRm = Rm(:,:,2);
I_f(:,:,2) = reshape(tmpRm(c_result_idx), size(Rv));
tmpRm = Rm(:,:,3);
I_f(:,:,3) = reshape(tmpRm(c_result_idx), size(Rv));

%final cleanup
I_f = I_f(L+1:end-L-P_R1, L+1:end-L-P_R1,:);  %removes the padded area from the filtered image
I_f = im2uint8(I_f);                          %converts image back to uint8

end