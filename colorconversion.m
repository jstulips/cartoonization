% COLORCONVERSION Converting Image to a Different Colorspace
    
function output = colorconversion(B, cspace)

switch lower(cspace)
    case 'lab'
        output = colorspace('Lab<-RGB',B);  % original conversion
    case 'luv'
        output = colorspace('LUV<-RGB',B);  % original conversion
    case 'lch'
        output = colorspace('LCH<-RGB',B);  % original conversion
    
    otherwise
end
    
    




