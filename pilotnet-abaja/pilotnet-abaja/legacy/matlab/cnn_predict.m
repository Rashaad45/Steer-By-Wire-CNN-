function y = cnn_predict(net, img)
%CNN_PREDICT Run PilotNet on one 66x200x3 uint8 RGB image -> row vector of heads.
switch net.backend
    case 'native'
        y = pilotnet_native_forward(net, img);
    case 'python'
        % H x W x 3 (column-major) -> row-major (H,W,C) byte order expected by numpy
        buf = reshape(permute(img, [3 2 1]), 1, []);
        r = net.obj.predict_bytes(py.bytes(buf), int32(size(img,1)), int32(size(img,2)));
        y = cellfun(@double, cell(r));
    case 'matlab'
        if strcmp(net.kind, 'dlnetwork')
            X = dlarray(single(img), 'SSCB');
            y = double(extractdata(predict(net.obj, X)));
        else
            y = double(predict(net.obj, single(img)));
        end
        y = reshape(y, 1, []);
end
end
