from diagrams import Cluster, Diagram
from diagrams.azure.compute import AppServices
from diagrams.aws.database import ElastiCache, RDS
from diagrams.aws.network import ELB
from diagrams.aws.network import Route53
from diagrams.generic.os import Ubuntu
from diagrams.generic.device import Tablet
from diagrams.programming.language import Python, Java

with Diagram("Protocol Risk Tool architecture", show=False):

    browser = Tablet("User browser")
    dash = Python("Plotly Dash\nfront end\nand business logic\n(Python)")
    tika = Java("Tika app service\nfor processing PDFs\n(Java)")

    browser >> dash >> tika
