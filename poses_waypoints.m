function q = poses_waypoints(robotName, poseName)
robotName = upper(robotName);
poseName  = upper(poseName);

switch robotName
    case 'UR3'
        switch poseName
            case 'HOME'
                qdeg = [0, 0, 0, 0, 0, 0];
            case 'POS3'
                qdeg = [42.7555608615, -36.7046325452, -88.4431694647, ...
                        -54.8521979901, -47.2430256994, 0];
            case 'POS4'
                qdeg = [103.7434853085, -67.2286817766, -28.5486608619, ...
                        -84.2226573615, -76.2565146915, 0];
            otherwise
                error('UR3 pose not found.');
        end

    case 'UR10E'
        switch poseName
            case 'HOME'
                qdeg = [-45, -45, 90, -45, 90, 0];
            case 'P1'
                qdeg = [-61.6773700249, -4.2703094749, -112.1683710003, ...
                         26.4386804752, 90, 0];
            case 'P2'
                qdeg = [-62.2077830829, -5.4850116729, -110.8300949769, ...
                         26.3151066498, 90, 0];
            case 'P3'
                qdeg = [-60.1874976551, -6.1401471190, -110.0956455397, ...
                         26.2357926516, 90, 0];
            case 'P4'
                qdeg = [-58.5628387530, -6.7499621958, -109.4040587612, ...
                         26.1540209571, 90, 0];
            otherwise
                error('UR10e pose not found.');
        end
    otherwise
        error('Unknown robot name.');
end

q = deg2rad(qdeg(:))';
end
