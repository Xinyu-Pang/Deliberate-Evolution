from typing import List, Dict, Any, Optional, Union
import time
import asyncio
import threading
from openai import OpenAI, AsyncOpenAI


class LLMClient:
    
    def __init__(
        self,
        api_type: str = "vllm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 10,
        retry_delay: float = 0.5,
        track_tokens: bool = True,
    ):
        self.api_type = api_type
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.track_tokens = track_tokens
        
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        
        self._token_lock = threading.Lock()
        
        self._init_client()
    
    def _init_client(self):
        if self.api_type == "openai":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
        elif self.api_type == "vllm":
            if self.base_url and ("openai" in self.base_url or "localhost" in self.base_url):
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
                self.async_client = AsyncOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            else:

                raise ValueError(f"Unsupported API type: {self.api_type}")
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")
    
    def call_openai_api(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        n: int = 1,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[str, List[str]]:
        if self.api_type != "openai":
            raise ValueError("call_openai_api only supports 'openai' api_type")
        
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        
        model = model or self.model
        if model is None:
            raise ValueError("Model must be specified")
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                n=n,
                temperature=temperature,
                stop=stop,
                extra_body=extra_body or {}
            )
            
            if self.track_tokens and hasattr(response, 'usage') and response.usage:
                self.completion_tokens += response.usage.completion_tokens
                self.prompt_tokens += response.usage.prompt_tokens
                self.total_tokens += response.usage.total_tokens
            
            if n == 1:
                content = response.choices[0].message.content
                return content.strip() if content else ""
            else:
                return [choice.message.content.strip() if choice.message.content else "" 
                        for choice in response.choices]
        
        except Exception as e:

            try:
                if 'response' in locals() and response.choices:
                    if response.choices[0].message.content is None:
                        return "" if n == 1 else [""] * n
            except:
                pass
            return "" if n == 1 else [""] * n
    
    
    
    def call_vllm_api(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.8,
        timeout: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        deterministic_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.api_type != "vllm":
            raise ValueError("call_vllm_api only supports 'vllm' api_type")
        
        if self.client is None:
            raise RuntimeError("vLLM client not initialized")
        
        model = model or self.model
        if model is None:
            raise ValueError("Model must be specified")
        
        if max_tokens is None:
            max_tokens = 8192
        
        params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        
        if deterministic_params:
            params.update(deterministic_params)
        
        if timeout is not None:
            params['timeout'] = timeout
        
        final_extra_body = extra_body or {}
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body=final_extra_body,
                    **params
                )
                
                if self.track_tokens and hasattr(completion, 'usage') and completion.usage:
                    self.completion_tokens += completion.usage.completion_tokens
                    self.prompt_tokens += completion.usage.prompt_tokens
                    self.total_tokens += completion.usage.total_tokens
                
                content = completion.choices[0].message.content
                return content.strip() if content else ""
            
            except Exception as e:
                last_error = e

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise RuntimeError(
                        f"Exceeded {self.max_retries} attempts in call_vllm_api"
                    ) from last_error
        
        raise RuntimeError("Unexpected error in call_vllm_api")
    
    async def async_call_vllm_api(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.8,
        timeout: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        deterministic_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.api_type != "vllm":
            raise ValueError("async_call_vllm_api only supports 'vllm' api_type")
        
        if self.async_client is None:
            raise RuntimeError("vLLM async client not initialized")
        
        model = model or self.model
        if model is None:
            raise ValueError("Model must be specified")
        
        if max_tokens is None:
            max_tokens = 8192
        
        params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        
        if deterministic_params:
            params.update(deterministic_params)
        
        if timeout is not None:
            params['timeout'] = timeout
        
        final_extra_body = extra_body or {}
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                completion = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body=final_extra_body,
                    **params
                )
                
                if self.track_tokens and hasattr(completion, 'usage') and completion.usage:
                    with self._token_lock:
                        self.completion_tokens += completion.usage.completion_tokens
                        self.prompt_tokens += completion.usage.prompt_tokens
                        self.total_tokens += completion.usage.total_tokens
                
                content = completion.choices[0].message.content
                return content.strip() if content else ""
            
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    raise RuntimeError(
                        f"Exceeded {self.max_retries} attempts in async_call_vllm_api"
                    ) from last_error
        
        raise RuntimeError("Unexpected error in async_call_vllm_api")
    
    async def async_call_llm(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        n: int = 1,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        deterministic_params: Optional[Dict[str, Any]] = None,
    ) -> Union[str, List[str]]:
        if self.api_type == "openai":
            if self.async_client is None:
                raise RuntimeError("OpenAI async client not initialized")
            
            model = model or self.model
            if model is None:
                raise ValueError("Model must be specified")
            
            if n > 1:
                tasks = [
                    self.async_client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=temperature or 0.0,
                        stop=stop,
                        extra_body=extra_body or {}
                    ) for _ in range(n)
                ]
                responses = await asyncio.gather(*tasks)
                
                if self.track_tokens:
                    with self._token_lock:
                        for response in responses:
                            if hasattr(response, 'usage') and response.usage:
                                self.completion_tokens += response.usage.completion_tokens
                                self.prompt_tokens += response.usage.prompt_tokens
                                self.total_tokens += response.usage.total_tokens
                
                return [choice.message.content.strip() if choice.message.content else "" 
                        for response in responses for choice in response.choices]
            else:
                response = await self.async_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature or 0.0,
                    stop=stop,
                    extra_body=extra_body or {}
                )

                
                if self.track_tokens and hasattr(response, 'usage') and response.usage:
                    with self._token_lock:
                        self.completion_tokens += response.usage.completion_tokens
                        self.prompt_tokens += response.usage.prompt_tokens
                        self.total_tokens += response.usage.total_tokens
                
                content = response.choices[0].message.content
                return content.strip() if content else ""
                
        elif self.api_type == "vllm":
            vllm_kwargs = {
                'messages': messages,
                'model': model,
                'temperature': temperature,
                'timeout': timeout,
                'extra_body': extra_body,
                'deterministic_params': deterministic_params
            }
            if max_tokens is not None:
                vllm_kwargs['max_tokens'] = max_tokens
            
            if n > 1:
                tasks = [self.async_call_vllm_api(**vllm_kwargs) for _ in range(n)]
                results = await asyncio.gather(*tasks)
                return results
            else:
                return await self.async_call_vllm_api(**vllm_kwargs)
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")
    
    def get_token_stats(self) -> Dict[str, int]:
        return {
            'completion_tokens': self.completion_tokens,
            'prompt_tokens': self.prompt_tokens,
            'total_tokens': self.total_tokens,
        }
    
    def reset_token_stats(self):
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
