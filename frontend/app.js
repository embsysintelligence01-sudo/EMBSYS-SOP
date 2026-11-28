'use strict';
const names={purity:'Purity Testing',packet:'Packet Sealing',releasing:'Packet Releasing'};
let token=sessionStorage.getItem('embsys_token')||'';
async function api(path,options={}) {
 const response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...(token?{'Authorization':'Bearer '+token}:{}),...options.headers}});
 if(response.status===401){document.querySelector('#authentication').hidden=false;throw Error('Enter the access token to connect.');}
 const data=await response.json();if(!response.ok)throw Error(data.detail||'Request failed');return data;
}
function text(selector,value){document.querySelector(selector).textContent=value;}
function description(a){return a?`${a.loan_id} · ${names[a.engine]}`:'None';}
async function refresh(){try{
 const data=await api('/api/state');text('#connection',data.state.replaceAll('_',' '));text('#current',description(data.current));text('#next',description(data.next));
 const body=document.querySelector('#queue');body.replaceChildren();const rows=data.queue.filter(a=>a.status!=='CANCELLED');
 document.querySelector('#empty').hidden=rows.length>0;
 for(const [index,a] of rows.entries()){
  const row=document.createElement('tr');for(const value of [index+1,a.loan_id,names[a.engine],a.status.replaceAll('_',' ')]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}
  const cell=document.createElement('td');if(a.can_remove){const button=document.createElement('button');button.className='remove';button.textContent='Remove';button.onclick=async()=>{try{await api('/api/queue/'+a.id,{method:'DELETE'});await refresh();}catch(e){text('#message',e.message);}};cell.append(button);}row.append(cell);body.append(row);
 }
 for(const box of document.querySelectorAll('[name=engine]')){box.disabled=!data.enabled.includes(box.value);box.parentElement.title=box.disabled?'Awaiting integration acceptance':'';}
 }catch(e){text('#connection','Disconnected');text('#message',e.message);}}
document.querySelector('#enqueue').onsubmit=async event=>{event.preventDefault();const engines=[...document.querySelectorAll('[name=engine]:checked')].map(e=>e.value);try{await api('/api/queue',{method:'POST',body:JSON.stringify({loan_id:document.querySelector('#loan').value.trim(),engines})});text('#message','Activities added to the queue.');await refresh();}catch(e){text('#message',e.message);}};
document.querySelector('#login').onclick=()=>{token=document.querySelector('#token').value;sessionStorage.setItem('embsys_token',token);document.querySelector('#authentication').hidden=true;refresh();};
refresh();setInterval(refresh,1500);
