from mcp.server.fastmcp import FastMCP
import os
server=FastMCP('Synthetic test server', host='127.0.0.1', port=int(os.environ.get('TEST_MCP_PORT', '8000')))
@server.tool()
def search(query:str)->dict:
    return {'result':[{'docid':f'd{i+1}','snippet':'Synthetic document text.','score':1.0}
                      for i in range(int(os.environ.get('TEST_MCP_K', '1')))]}
@server.tool()
def get_document(docid:str)->dict:
    return {'result':{'docid':docid,'text':'Synthetic document text.'}}
if __name__=='__main__': server.run(transport='streamable-http' if os.environ.get('TEST_MCP_PORT') else 'stdio')
