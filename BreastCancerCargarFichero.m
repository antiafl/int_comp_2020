%% Import data from text file.
% Script for importing data from the following text file:
%
%    E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\p0\breast-cancer-wisconsin.data
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2020/02/27 18:01:31

%% Initialize variables.
filename = 'E:\MASTER\CUATRI_2\2.4_Inteligencia_Computacional\Practicas\p0\breast-cancer-wisconsin.data';
delimiter = ',';

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%s%s%s%s%s%s%s%s%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric strings to numbers.
% Replace non-numeric strings with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3,4,5,6,7,8,9,10,11]
    % Converts strings in the input cell array to numbers. Replaced non-numeric
    % strings with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(thousandsRegExp, ',', 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric strings to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
breastcancerwisconsin = cell2mat(raw);

% Last column contains classes
OUTPUTS_num = breastcancerwisconsin(:,11);

% Leave out first column that contains irrelevant information (id number)
INPUTS = breastcancerwisconsin(:,2:10);
%% Calculamos la media de cada una de las columnas sin tener en cuenta los
%valores NaN

% Calculate mean of each column
mean_array = nanmean(INPUTS,1);    

%Reemplazamos los valores NaN por la media de la columna
[fila,columna] = find(isnan(INPUTS));
for i=1:size(fila)    
    for j=1:size(columna)
        INPUTS(fila(i),columna(j)) = mean_array(columna(j));
    end
end

%% Cambiar valores de OUTPUTS_cancer de num�rico a texto para mayor inteligibilidad
% 2 clases: tumor benigno -> valor 2, y tumor maligno -> valor 4
for i=1:size(OUTPUTS_num)
    switch OUTPUTS_num(i)
        case 2 
            OUTPUTS{i,1} = 'BENIGNO';
        case 4            
            OUTPUTS{i,1} = 'MALIGNO';
    end
end
[NumData, Attributes] = size(INPUTS);
[NumClass, c] = size(unique(OUTPUTS));
save('cancerVars','INPUTS','OUTPUTS','NumData','NumClass','OUTPUTS_num');
%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me R mean_array fila columna i j Attributes c breastcancerwisconsin;