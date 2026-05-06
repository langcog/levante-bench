
const handler = require('./api/results-report.js');
const req = { method: 'GET', query: {} };
const res = {
  statusCode: 200,
  headers: {},
  setHeader(k,v){ this.headers[k]=v; },
  status(c){ this.statusCode=c; return this; },
  send(body){
    console.log('STATUS', this.statusCode);
    if (typeof body === 'string') {
      console.log(body.slice(0, 2000));
    } else {
      console.log(String(body).slice(0, 2000));
    }
  }
};
handler(req,res).catch(err=>{ console.error('ERR', err && err.stack || err); process.exit(1); });
