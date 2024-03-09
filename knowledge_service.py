#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
vector service
"""

import os


from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from rapidocr_onnxruntime import RapidOCR


class KnowledgeService(object):

    def __init__(self):
        self.knowledge_base = None
        self.docs_path = '/kaggle/ChatGLM3/docs/'
        self.knowledge_base_path = '/kaggle/ChatGLM3/knowledge_base/'
        self.embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')

    def init_knowledge_base(self):
        """
        初始化本地知识库向量
        """
        print('\n#####init_knowledge_base#####\n')
        docs = []
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=20)
        for doc in os.listdir(self.docs_path):
            if doc.endswith('.txt'):
                print(doc)
                loader = UnstructuredFileLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                split_doc = text_splitter.split_documents(doc)
                docs.extend(split_doc)
            elif doc.endswith('.md'):
                print(doc)
                loader = UnstructuredMarkdownLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                split_doc = markdown_splitter.split_documents(doc)
                docs.extend(split_doc)
            elif doc.endswith('.pdf'):
                print(doc)
                loader = UnstructuredPDFLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                split_doc = markdown_splitter.split_documents(doc)
                docs.extend(split_doc)
            elif doc.endswith('.jpg'):
                print(doc)
                ocr = RapidOCR()
                result, _ = ocr(f'{self.docs_path}/{doc}')
                img_docs = ""
                if result:
                    ocr_result = [line[1] for line in result]
                    img_docs += "\n".join(ocr_result)
                split_docs = text_splitter.create_documents([img_docs])
                docs.extend(split_docs)

        # 这里调用出问题，
        self.knowledge_base = FAISS.from_documents(docs, self.embeddings)

        self.save_knowledge_base()

    def save_knowledge_base(self):
        # 确保有一个目录来保存知识库索引
        if not os.path.exists(self.knowledge_base_path):
            os.makedirs(self.knowledge_base_path)

        # 知识库索引文件的完整路径
        index_file_path = os.path.join(self.knowledge_base_path, 'knowledge_base.index')

        if isinstance(self.knowledge_base, faiss.Index):
            # 保存知识库索引
            faiss.write_index(self.knowledge_base, index_file_path)
            print(f"知识库索引已保存到 {index_file_path}")
        else:
            print("无法保存索引：self.knowledge_base 不是一个 FAISS Index 实例。")

    def add_document(self, document_path):
        split_doc = []
        if document_path.endswith('.txt'):
            print(document_path)
            loader = UnstructuredFileLoader(document_path, mode="elements")
            doc = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = text_splitter.split_documents(doc)
        elif doc.endswith('.md'):
            print(document_path)
            loader = UnstructuredMarkdownLoader(document_path, mode="elements")
            doc = loader.load()
            markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = markdown_splitter.split_documents(doc)
        elif doc.endswith('.pdf'):
            print(document_path)
            loader = UnstructuredPDFLoader(document_path, mode="elements")
            doc = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = text_splitter.split_documents(doc)
        elif doc.endswith('.jpg'):
            print(document_path)
            loader = UnstructuredPDFLoader(document_path, mode="elements")
            docs = self.init_knowledge_base(jpg_file)
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = text_splitter.create_documents([docs])

        if not self.knowledge_base:
            self.knowledge_base = FAISS.from_documents(split_doc, self.embeddings)
        else:
            self.knowledge_base.add_documents(split_doc)

    #  path=None本来是没有的，但是为了便于理解path的凭空产生，所以加了一个
    def load_knowledge_base(self, path=None):
        # 如果没有提供路径，则使用默认的知识库索引文件路径
        index_file_path = os.path.join(self.knowledge_base_path, 'knowledge_base.index') if path is None else path

        # 加载知识库索引
        if os.path.exists(index_file_path):
            self.knowledge_base = faiss.read_index(index_file_path)
            print(f"已从 {index_file_path} 加载知识库索引。")
        else:
            print("知识库索引文件不存在，将初始化一个新的知识库索引。")
            self.init_knowledge_base()

        return self.knowledge_base




