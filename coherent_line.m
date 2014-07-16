% Coherent line drawing function detecting edges and lines on the image following the method outlined in:
%
% Kang, H., Lee, S., & Chui, C. K. (2007). 
% Coherent line drawing. Proceedings of the 5th international symposium on
% Non-photorealistic animation and rendering, San Diego, California,43–50. 
% DOI: 10.1145/1274871.1274878
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function im_result = coherent_line(im_color)

if ndims(im_color) < 3
    im_gray = im_color;
else
    im_gray = double(rgb2gray(im_color));
end

%[G_x G_y] = gradient(im_gray);
im_gray = imfilter(im_gray, fspecial('gaussian',[5 5]));
G_x = imfilter(im_gray, fspecial('sobel')','replicate');
G_y = imfilter(im_gray, fspecial('sobel'),'replicate');
G_x = double(G_x); G_y = double(G_y);
G = sqrt(G_x.^2+G_y.^2);
figure; imshow(G); title('Bi-directional Sobel gradient');
t_x = G_y;
t_y = -G_x;
k = sqrt(t_x.^2+t_y.^2);
t_x(k~=0) = t_x(k~=0)./k(k~=0);
t_y(k~=0) = t_y(k~=0)./k(k~=0);

mu = 5; % box filter radius

niterations = 3; % 2 or 3

t_x_ver = t_x;
t_y_ver = t_y;
t_x_hor = t_x;
t_y_hor = t_y;

% separated edge tangent field
for iteration = 1:niterations
    [t_x_hor t_y_hor] = separated_etf_horizontal(t_x_ver, t_y_ver, G, mu);
    [t_x_ver t_y_ver] = separated_etf_vertical(t_x_hor, t_y_hor, G, mu);
    t_x = t_x_ver;
    t_y = t_y_ver;
end

sigma_m = 3.0;
sigma_c = 1.0;
rho = 0.99;
tau = 1;

S = 5; % sample points along c_x
T = 3; % sample points along l_s

delta_m = 1;
delta_n = 1;

% gradient direction
G_x = t_y;
G_y = -t_x;

for iter=1:1
    disp(['FDOG Iteration #',num2str(iter),'...']);

    % optional: Gaussian-blur input image to smooth out line strokes
    im_gray = imfilter(im_gray, fspecial('gaussian',[5 5])); 
    
    H_g = separated_fdog_gradient(im_gray, G_x, G_y, sigma_c, T, rho, delta_n);     
    H_e = separated_fdog_tangent(H_g, t_x, t_y, sigma_m, S, delta_m);

    %im_result = H_e>0;
    im_result = H_e>0 & 1+tanh(H_e)>tau;
    %im_showresult = imfilter(double(im_result),fspecial('gaussian',[3 3]));
    figure, imshow(im_result);
    
    % combine back edge with original image & ensure it is grayscale
    if ndims(im_color) < 3
        im_gray = im_result.*double(im_color);
    else
       im_color_rev = repmat(im_result,[1 1 3]).*double(im_color);
       im_gray = double(rgb2gray(im_color_rev));
    end
end

% FUNCTIONS
% *************************************************************************
%==========================================================================
function [t_x_hor t_y_hor] = separated_etf_horizontal(t_x_ver, t_y_ver, G, mu)
t_x_hor = t_x_ver/2;
t_y_hor = t_y_ver/2;
for d = 1:mu
    dot = t_x_ver(:,1:end-d).*t_x_ver(:,d+1:end)+t_y_ver(:,1:end-d).*t_y_ver(:,d+1:end);
    diff = G(:,d+1:end)-G(:,1:end-d);
    w_d = [dot zeros(size(G,1),d)];
    w_m = [(diff+1)/2 zeros(size(G,1),d)];
    t_x_hor = t_x_hor+w_d.*w_m.*t_x_ver;
    t_y_hor = t_y_hor+w_d.*w_m.*t_y_ver;
    w_d = [zeros(size(G,1),d) dot];
    w_m = [zeros(size(G,1),d) (-diff+1)/2];
    t_x_hor = t_x_hor+w_d.*w_m.*t_x_ver;
    t_y_hor = t_y_hor+w_d.*w_m.*t_y_ver;
end
k = sqrt(t_x_hor.^2+t_y_hor.^2);
t_x_hor(k~=0) = t_x_hor(k~=0)./k(k~=0);
t_y_hor(k~=0) = t_y_hor(k~=0)./k(k~=0);


function [t_x_ver t_y_ver] = separated_etf_vertical(t_x_hor, t_y_hor, G, mu)
t_x_ver = t_x_hor/2;
t_y_ver = t_y_hor/2;
for d = 1:mu
    dot = t_x_ver(1:end-d,:).*t_x_ver(d+1:end,:)+t_y_ver(1:end-d,:).*t_y_ver(d+1:end,:);
    diff = G(d+1:end,:)-G(1:end-d,:);
    w_d = [dot; zeros(d,size(G,2))];
    w_m = [(diff+1)/2; zeros(d,size(G,2))];
    t_x_hor = t_x_hor+w_d.*w_m.*t_x_ver;
    t_y_hor = t_y_hor+w_d.*w_m.*t_y_ver;
    w_d = [zeros(d,size(G,2)); dot];
    w_m = [zeros(d,size(G,2)); (-diff+1)/2;];
    t_x_hor = t_x_hor+w_d.*w_m.*t_x_ver;
    t_y_hor = t_y_hor+w_d.*w_m.*t_y_ver;
end
k = sqrt(t_x_ver.^2+t_y_ver.^2);
t_x_ver(k~=0) = t_x_ver(k~=0)./k(k~=0);
t_y_ver(k~=0) = t_y_ver(k~=0)./k(k~=0);


% difference of gaussians along gradient direction
function im_result = separated_fdog_gradient(im_gray, G_x, G_y, sigma_c, T, rho, delta_n)
sigma_s = 1.6*sigma_c;
q = T*2+1;
[pos_x pos_y] = meshgrid(1:size(im_gray,2), 1:size(im_gray,1));
kernel_c = exp(-(-T:T).^2/(2*sigma_c^2))/(sqrt(2*pi)*sigma_c); % center
kernel_s = exp(-(-T:T).^2/(2*sigma_s^2))/(sqrt(2*pi)*sigma_s); % surrounding
kernel_q = kernel_c/sum(kernel_c)-rho*kernel_s/sum(kernel_s);
H_pos_x = zeros(size(im_gray,1), size(im_gray,2), q);
H_pos_y = zeros(size(im_gray,1), size(im_gray,2), q);

H_g = im_gray*kernel_q(T+1);
H_pos_x(:,:,T+1) = pos_x;
H_pos_y(:,:,T+1) = pos_y;
for t = 1:T
    tp = t+T+1;
    out_of_bound = H_pos_x(:,:,tp-1)+G_x*delta_n>size(im_gray,2) | H_pos_x(:,:,tp-1)+G_x*delta_n<1 | ...
        H_pos_y(:,:,tp-1)+G_y*delta_n>size(im_gray,1) | H_pos_y(:,:,tp-1)+G_y*delta_n<1;
    H_pos_x(:,:,tp) = H_pos_x(:,:,tp-1)+G_x*delta_n.*~out_of_bound;
    H_pos_y(:,:,tp) = H_pos_y(:,:,tp-1)+G_y*delta_n.*~out_of_bound;
    H_g = H_g+interp2(pos_x, pos_y, im_gray, H_pos_x(:,:,tp), H_pos_y(:,:,tp))*kernel_q(tp);
    tn = T+1-t;
    out_of_bound = H_pos_x(:,:,tn+1)-G_x*delta_n>size(im_gray,2) | H_pos_x(:,:,tn+1)-G_x*delta_n<1 | ...
        H_pos_y(:,:,tn+1)-G_y*delta_n>size(im_gray,1) | H_pos_y(:,:,tn+1)-G_y*delta_n<1;
    H_pos_x(:,:,tn) = H_pos_x(:,:,tn+1)-G_x*delta_n.*~out_of_bound;
    H_pos_y(:,:,tn) = H_pos_y(:,:,tn+1)-G_y*delta_n.*~out_of_bound;
    H_g = H_g+interp2(pos_x, pos_y, im_gray, H_pos_x(:,:,tn), H_pos_y(:,:,tn))*kernel_q(tn);
end
im_result = H_g;


% gaussian along tangent axis
function im_result = separated_fdog_tangent(im_gray, t_x, t_y, sigma_m, S, delta_m)
p = S*2+1;
[pos_x pos_y] = meshgrid(1:size(im_gray,2), 1:size(im_gray,1));
H_e_p = zeros(size(im_gray, 1), size(im_gray, 2), p);
H_pos_x = zeros(size(H_e_p));
H_pos_y = zeros(size(H_e_p));
H_weight = zeros(size(im_gray));
kernel_p = exp(-(-S:S).^2/(2*sigma_m^2))/(sqrt(2*pi)*sigma_m);
kernel_p = kernel_p/sum(kernel_p);
H_pos_x(:,:,S+1) = pos_x;
H_pos_y(:,:,S+1) = pos_y;
H_e_p(:,:,S+1) = im_gray*kernel_p(S+1);
H_weight = H_weight+kernel_p(S+1);
for s = 1:S
    % positive direction
    sp = s+S+1;
    H_pos_x(:,:,sp) = H_pos_x(:,:,sp-1)+interp2(pos_x, pos_y, t_x, H_pos_x(:,:,sp-1), H_pos_y(:,:,sp-1), 'linear', 0)*delta_m;
    H_pos_y(:,:,sp) = H_pos_y(:,:,sp-1)+interp2(pos_x, pos_y, t_y, H_pos_x(:,:,sp-1), H_pos_y(:,:,sp-1), 'linear', 0)*delta_m;
    tmp = interp2(pos_x, pos_y, im_gray, H_pos_x(:,:,sp), H_pos_y(:,:,sp));
    H_weight = H_weight+kernel_p(sp)*~isnan(tmp);
    tmp(isnan(tmp)) = 0;
    H_e_p(:,:,sp) = tmp.*kernel_p(sp);
    
    % negative direction
    sn = S+1-s;
    H_pos_x(:,:,sn) = H_pos_x(:,:,sn+1)+interp2(pos_x, pos_y, t_x, H_pos_x(:,:,sn+1), H_pos_y(:,:,sn+1), 'linear', 0)*delta_m;
    H_pos_y(:,:,sn) = H_pos_y(:,:,sn+1)+interp2(pos_x, pos_y, t_y, H_pos_x(:,:,sn+1), H_pos_y(:,:,sn+1), 'linear', 0)*delta_m;
    tmp = interp2(pos_x, pos_y, im_gray, H_pos_x(:,:,sn), H_pos_y(:,:,sn))*kernel_p(sn);
    H_weight = H_weight+kernel_p(sn)*~isnan(tmp);
    tmp(isnan(tmp)) = 0;
    H_e_p(:,:,sn) = tmp.*kernel_p(sn);
end
im_result = sum(H_e_p, 3);
