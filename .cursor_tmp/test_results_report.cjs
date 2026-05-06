const handler = require('/home/david/levante/levante-bench/api/results-report.js');
const req = { method: 'GET', query: {} };
const res = {
  statusCode: 200,
  headers: {},
  setHeader(k,v){ this.headers[k]=v; },
  status(c){ this.statusCode=c; return this; },
  send(body){
    console.log('STATUS', this.statusCode);
    const text = typeof body === 'string' ? body : JSON.stringify(body);
    console.log(text.slice(0, 1500));
  }
};
Promise.resolve(handler(req, res)).catch((e)=>{
  console.error('ERR', e && e.message ? e.message : e);
  process.exit(1);
});
