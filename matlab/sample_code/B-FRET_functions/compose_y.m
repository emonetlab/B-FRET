function y = compose_y(data)
IDD = data.IDD;
IDA = data.IDA;
y = cell(1,length(IDD));

% composing y
for k = 1:length(IDD)
    y{k} = [IDD(k);IDA(k)];
end
end

