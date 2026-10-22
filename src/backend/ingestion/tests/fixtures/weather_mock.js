// Contract fixture based on Open-Meteo's documented Single Runs API.
const http = require('http');
const attempts = new Map();

http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const run = url.searchParams.get('run');
  if (!run || url.searchParams.get('models') !== 'ecmwf_ifs'
      || url.searchParams.get('hourly') !== 'precipitation'
      || url.searchParams.get('timezone') !== 'UTC'
      || url.searchParams.get('timeformat') !== 'unixtime'
      || url.searchParams.get('precipitation_unit') !== 'mm') {
    res.writeHead(400, {'Content-Type': 'application/json'});
    res.end(JSON.stringify({error: true, reason: 'Invalid forecast parameters'}));
    return;
  }
  if (url.pathname.endsWith('/unavailable')
      || (url.pathname.endsWith('/fallback') && run.endsWith('06:00'))) {
    res.writeHead(404, {'Content-Type': 'application/json'});
    res.end(JSON.stringify({error: true, reason: 'Run not available'}));
    return;
  }
  if (url.pathname.endsWith('/html')) {
    res.writeHead(502, {'Content-Type': 'text/html'});
    res.end('<html>Upstream unavailable</html>');
    return;
  }
  const retryKey = `${run}:${req.headers['x-mock-session'] || ''}`;
  if (url.pathname.endsWith('/retry') && !attempts.has(retryKey)) {
    attempts.set(retryKey, 1);
    res.writeHead(429, {'Content-Type': 'application/json', 'Retry-After': '1'});
    res.end(JSON.stringify({error: true, reason: 'Too many requests'}));
    return;
  }
  const cycle = Date.parse(`${run}:00Z`) / 1000;
  const times = Array.from({length: 73}, (_, i) => cycle + i * 3600);
  const precipitation = times.map((_, i) => i === 0 ? null : 0);
  if (url.pathname.endsWith('/invalid')) precipitation[3] = -1;
  res.writeHead(200, {'Content-Type': 'application/json'});
  res.end(JSON.stringify({
    latitude: 16.221111,
    longitude: 107.179444,
    utc_offset_seconds: 0,
    hourly_units: {time: 'unixtime', precipitation: 'mm'},
    hourly: {time: times, precipitation},
  }));
}).listen(process.env.PORT || 4900, '127.0.0.1');
