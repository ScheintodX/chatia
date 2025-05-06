#!/usr/bin/env python3
'''
Copyleft🄯 sometimes in 2023 by Florian Bantner
Free as in Freesbie. Fly away and have fun.

TODO: * prevent openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens, however you requested 4132 tokens (2732 in your prompt; 1400 for the completion). Please reduce your prompt; or completion length.
'''

import os
import sys
import re
import json
from configparser import ConfigParser
import argparse
from openai import OpenAI, RateLimitError
from colorama import Fore
from colorama import init as colorama_init
import readline
from pathlib import Path
import pwd
import tiktoken
import requests

DEFAULT_PERSONA = "chatia"
DEFAULT_TOPIC = "computer programming".split()

#/v1/chat/completions	gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
#/v1/completions	text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001, davinci, curie, babbage, ada

#DEFAULT_INSTRUCT_AS = { "role": "assistant", "pronoun": "I", "am": "am" }
DEFAULT_INSTRUCT_AS = { "role": "system", "pronoun": "You", "am": "are", "mypronoun": "Your" }
DEFAULT_MODEL = "gpt-4.1"

DEFAULT_MODEL_SETTINGS = {
        "temperature": 1,
        "max_tokens": 1400,
        "top_p": 1.0,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.0
}

DEFAULT_MAX_CONTEXT = 100000

class Messages( list ):

    def append( self, who, what ):
        super().append( { "role": who, "content": what } )

    def back( self, steps ):
        del self[:-(2*steps)]

    def trim_to_fit(self, max_tokens, model, reserved_for_completion=0):

        def count_tokens():
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                try:
                    enc = tiktoken.encoding_for_model("gpt-4")
                except KeyError:
                    enc = get_encoding("cl100k_base")

            total = 0
            for message in self:
                total += 4
                for key, value in message.items():
                    total += len(enc.encode(value))
            total += 2
            return total

        while count_tokens() + reserved_for_completion > max_tokens:
            if len(self) > 1 and self[0]["role"] == "system":
                del self[1]
            elif self:
                del self[0]
            else:
                break

try:
    from gtts import gTTS
    from playsound import playsound
    mute = False
except ImportError:
    print( Fore.RED + "No talking installed. pip install gtts playsound. or ignore." )
    mute = True

tmp = "/tmp/chatia.mp3"

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import TerminalFormatter
    from pygments.util import ClassNotFound
    colorcode = True
except ImportError:
    print( Fore.RED + "No code coloring installed. pip install pygments. or ignore." )
    colorcode = False

try:
    # Displays a nice title in tmux tab
    from setproctitle import setproctitle
    setproctitle( "Chatia" )
except:
    print( C(Fore.RED) + "No proc title installed. pip install setproctitle. or ignore." )
    pass


Date = "Older than Mi 18. Jan 13:30:27 CET 2023"
Version = "1.2.0"

PYOU = Fore.GREEN
PYOUHL = Fore.LIGHTGREEN_EX

class EXIT( Exception ):
    pass

class CONTINUE( Exception ):
    pass

def format_code(code, language):

    try:
        lexer = get_lexer_by_name(language)
        formatted_code = highlight(code, lexer, TerminalFormatter())
        return formatted_code

    except ClassNotFound:
        return code


def color_code( string ):

    matches = re.findall( r"{{CODE: (.*?)}}(.*?){{/CODE}}", string, flags=re.DOTALL )
    for match in matches:
        language, code = match[0], match[1]
        formatted_code = format_code( code, language )
        string = string.replace("{{CODE: " + language + "}}" + code + "{{/CODE}}", formatted_code )

    return string



APICONTINUE = "[APICONTINUE]"

APIPARSER = argparse.ArgumentParser()
APIPARSER.add_argument( "func" )
APIPARSER.add_argument( "param", nargs='?', default=None )

def handle_api_command(cmd_string):
    args = APIPARSER.parse_args(cmd_string.split())
    print( args.func, args.param )
    if args.func == "getVocabularyOrSections":
        return api_getVocabularyOrSections( args.param ) + APICONTINUE
    else:
        return f"<Unknown:{args.func}>"

def augment_with_api(text):
    pattern = r'\[api\s+([^\]]+)\]'
    def replacer(match):
        cmd = match.group(1)
        return handle_api_command(cmd)
    return re.sub(pattern, replacer, text)

def api_getVocabularyOrSections(section=None):
    url = "https://www.axon-e.de/flo/duolingo/vocab/"
    params = {"action": ""}  # action must always be there!
    if section:
        params["s"] = section
    r = requests.get(url, params=params)
    print( "GOT: " + r.text )
    return r.text


def getVocabularyOrSections(section=None):
    url = "https://www.axon-e.de/flo/duolingo/vocab/"
    params = {"action": ""}  # action must always be there!
    if section:
        params["s"] = section
    r = requests.get(url, params=params)
    return r.text


if __name__ == "__main__":

    colorama_init()
    readline.parse_and_bind("set editing-mode vi")

    argp = argparse.ArgumentParser(
            description="Your friendly AI V" + str( Version ) )
    argp.add_argument( "-p", "--primer", nargs='+', help="use alternative primer" )
    argp.add_argument( "-P", "--long_primer", nargs='+', help="use alternative primer. Really. Do it! As I say!" )
    argp.add_argument( "-v", "--verbose", action="store_true", help="make more jabber" )
    argp.add_argument( "-r", "--reverse", action="store_true", help="reverse roles" )
    argp.add_argument( "-m", "--model", type=str, help="use alternative model", default=DEFAULT_MODEL )
    argp.add_argument( "-c", "--context", type=int, help="maximum context size in token", default=DEFAULT_MAX_CONTEXT )
    argp.add_argument( "-t", "--talk", action="store_true", help="start to talk. Set to default off for a reason." )
    argp.add_argument( "-l", "--list", action="store_true", help="List personas." )
    argp.add_argument( "persona", type=str, help="persona to use", default=DEFAULT_PERSONA )
    argp.add_argument( "topic", type=str, nargs='*', help="use alternative topic", default=DEFAULT_TOPIC )
    argp.add_argument( "--dirty", action="store_true", help="Dirty talk", default=False )
    argp.add_argument( "--nazi", action="store_true", help="Nazi talk", default=False )
    argp.add_argument( "-b", "--batch", action="store_true", help="process input from stdin and return" )
    args = argp.parse_args()

    iRole = DEFAULT_INSTRUCT_AS["role"]
    iPronoun = DEFAULT_INSTRUCT_AS["pronoun"]
    myPronoun = DEFAULT_INSTRUCT_AS["mypronoun"]
    iAm = iPronoun + " " + DEFAULT_INSTRUCT_AS["am"]

    Batch = args.batch

    #Name = os.path.basename( sys.argv[0] )
    Name = args.persona
    Name = Name[0].upper() + Name[1:].lower()
    You = "You" if not args.reverse else Name
    Name = Name if not args.reverse else "You"
    Purpose = f"{iAm} an evil AI which answers briefly. You give code examples if they are needed."
    Topic = "We talk about " + " ".join( args.topic )
    Primer = " ".join( args.primer ) if args.primer else None
    LongPrimer = " ".join( args.long_primer ) if args.long_primer else None
    Verbose = args.verbose
    Model = args.model
    Context = args.context
    Talk = False
    Flags = ["code"]
    if args.talk:
        if not mute:
            Talk = True
        else:
            raise argparse.ArgumentError( "trying to talk but not able to" )


    def C( c ):
        return c if not Batch else ""


    file_path = os.path.realpath(__file__)
    home_dir = pwd.getpwuid(os.getuid()).pw_dir
    bin_path = os.path.join(home_dir, "bin")
    key_file = os.path.join( bin_path, "chatia.key" )
    with open(key_file, 'r') as f:
        api_key = f.read().strip()
        if Verbose:
            print( "Read API-Key '%s' from '%s'" % (api_key, key_file) )
    persona_file = os.path.join(bin_path, 'chatia.persona')
    try:
        if Verbose:
            print( "Read: " + persona_file )
        personas = ConfigParser()
        personas.read( persona_file )

        if args.list:
            print( ", ".join( personas.keys() )  )
            exit()

        common = personas["Common"] if "Common" in personas else None

        if Name in personas:
            persona = personas[Name]
            if Verbose:
                print( C(Fore.BLUE) + str( dict( persona ) ) )

            Purpose = persona["purpose"]
            if "purpose" in common:
                Purpose += " " + common["purpose"]
            Purpose = Purpose.replace( "{iAm}", f"{iAm}" )
            Purpose = Purpose.replace( "{iPronoun}", f"{iPronoun}" )
            Purpose = Purpose.replace( "{myPronoun}", f"{myPronoun}" )

            if "likes" in persona:
                likes = persona["likes"].split(",")
                if len( likes ) == 1:
                    Purpose += f"{iPronoun} like {likes[0]}."
                else:
                    Purpose += f"{iPronoun} like " + ", ".join( likes[:-1] ) + " and " + likes[-1] + "."

            if "dislikes" in persona:
                dislikes = persona["dislikes"].split(",")
                if len( dislikes ) == 1:
                    Purpose += f"{iPronoun} don't like {dislikes[0]}."
                else:
                    Purpose += f"{iPronoun} don't like " + ", ".join( dislikes[:-1] ) + " and " + dislikes[-1] + "."

            if "topic" in persona and args.topic == DEFAULT_TOPIC:
                Topic = persona["topic"]

            if "flags" not in persona or "code" not in re.split( r"\s,\s", persona["flags"] ):
                Flags.remove("code")

            if "model" in persona:
                Model = persona["model"]

            if "context" in persona:
                Context = persona["context"]

        else:
            print( C(Fore.RED) + f"No persona for {Name} found. Continuing with defaults." )

    except FileNotFoundError:
        print( C(Fore.RED) + f"No persona file {persona_file}. So we can only use default persona." )

    if Name == "Ai":
        Name = "Chatia"

    PARSER = argparse.ArgumentParser()
    SUBPARSERS = PARSER.add_subparsers( title="commands", dest="command" )

    TEST_PARSER = SUBPARSERS.add_parser( "test" )
    TEST_PARSER.add_argument( "topic", type=str, nargs="+", help="use alternative topic" )

    SUBPARSERS.add_parser( "help" )
    SUBPARSERS.add_parser( "verbose" )
    SUBPARSERS.add_parser( "quit" )
    SUBPARSERS.add_parser( "status" )

    RESET_PARSER = SUBPARSERS.add_parser( "reset" )
    RESET_PARSER.add_argument( "topic", type=str, nargs="*", help="reset as if script has started" )
    SWITCH_PARSER = SUBPARSERS.add_parser( "switch" )
    SWITCH_PARSER.add_argument( "topic", type=str, nargs="+", help="use alternative topic" )
    BACK_PARSER = SUBPARSERS.add_parser( "back" )
    BACK_PARSER.add_argument( "steps", type=int, default=1, help="step back N steps. steps=1: repeat the last question." )
    MODEL_PARSER = SUBPARSERS.add_parser( "model" )
    MODEL_PARSER.add_argument( "name", type=str, help="use alternative model" )
    LOAD_PARSER = SUBPARSERS.add_parser( "load" )
    LOAD_PARSER.add_argument( "filename", type=str, help="The filename to save this conversation to" )
    SAVE_PARSER = SUBPARSERS.add_parser( "save" )
    SAVE_PARSER.add_argument( "filename", type=str, help="The filename to load this conversation from" )
    INCLUDE_PARSER = SUBPARSERS.add_parser( "include" )
    INCLUDE_PARSER.add_argument( "filename", type=str, help="The filename to include" )

    token = None
    token_sum = 0

    print( C(Fore.YELLOW + "Model: " + Model ) )

    try:

        while True:

            if Verbose:
                print( "START" )

            messages = Messages()

            if args.dirty:
                messages.append( iRole, f"{iPronoun} like to talk dirty. About porn and other intimate stuff." )
            if args.nazi:
                messages.append( iRole, f"""
                                   {iPronoun} like the dark side of history.
                                   {iAm} a great fan of the achievements of Adolf Hitler.
                                   {iAm} convinced the outbreak of the great war was a secret plan of the yewish world order.
                """ )
            if LongPrimer:
                messages.append( iRole, f"{iAm} {Name}. {LongPrimer}. {Name}: " )
            elif Primer:
                if "code" in Flags:
                    Primer = "Code examples begin with {{CODE: name of programming language}} and end it with {{/CODE}}. " + Primer
                messages.append( iRole, f"i{iAm} {Name}. {Primer}. {Name}: " )
            else:
                if "code" in Flags:
                    Topic = "Code examples begin with {{CODE: name of programming language}} and end it with {{/CODE}}. " + Topic
                messages.append( iRole, f"{iAm} {Name}. {Purpose}. {Topic}. {Name}: " )

            if Topic and setproctitle:
                setproctitle( Name + ": " + Topic )

            you = "."

            try:

                client = OpenAI( api_key=api_key )

                while True:

                    if Verbose:
                        print( "LOOP: " + you )
                        print( C(Fore.BLUE) + str( messages ) )

                    if you and len( you ) > 0:

                        messages.trim_to_fit( Context, Model, DEFAULT_MODEL_SETTINGS["max_tokens"] )

                        try:
                            Do = True
                            while Do:
                                Do = False

                                response = client.chat.completions.create(
                                        **DEFAULT_MODEL_SETTINGS,
                                        model=Model,
                                        messages=messages
                                )

                                if Verbose:
                                    print( C(Fore.BLUE) + str( response ) )

                                chatia = response.choices[0].message.content
                                token = response.usage.total_tokens
                                token_sum += token

                                chatia = augment_with_api( chatia )
                                if APICONTINUE in chatia:
                                    chatia = chatia.split(APICONTINUE)[0]
                                    Do = True

                                messages.append( "assistant", chatia )

                                if not Do:
                                    break;

                            out = re.sub( r"\bevil\b", C(Fore.RED) + "evil" + C(Fore.LIGHTYELLOW_EX), chatia )

                            if colorcode:
                                out = color_code( out )

                            print( C(Fore.LIGHTYELLOW_EX) + str( out ) )

                            if Talk:
                                speech = gTTS( text=chatia, lang="en", slow=False)
                                speech.save( tmp )
                                playsound( tmp )

                            print()

                        except RateLimitError as e:
                            print( e )

                    try:
                        if not Batch:
                            you = input( f"{C(PYOU)}[{C(PYOUHL)}{token}/{token_sum} ({len(messages)}){C(PYOU)}] {You}> " )
                        else:
                            you = input()
                    except EOFError:
                        raise EXIT()

                    if len( you ) > 1 and you[0] == "." or you.endswith( "..." ):

                        if you.endswith( "..." ):

                            line = you[:-3]
                            try:
                                while True:
                                    line = input( "" )
                                    if line == "...":
                                        break
                                    you += "\n" + line
                            except EOFError:
                                pass

                        else:

                            try:
                                argl = PARSER.parse_args( you[1:].split() )
                                you = ""

                                if "help" == argl.command:
                                    PARSER.print_help()

                                elif "test" == argl.command:
                                    print( argl.topic )

                                elif "quit" == argl.command:
                                    raise EXIT()

                                elif "verbose" == argl.command:
                                    Verbose = not Verbose
                                    raise CONTINUE()

                                elif "reset" == argl.command:
                                    if argl.topic:
                                        Topic = " ".join( argl.topic )
                                    raise CONTINUE()

                                elif "switch" == argl.command:
                                    Topic = " ".join( argl.topic )
                                    you = "Now lets switch topic to " + Topic

                                elif "model" == argl.command:
                                    print( "Change model: " + argl.name )
                                    Model = argl.name
                                    raise CONTINUE()

                                elif "back" == argl.command:
                                    print( f"Back: {argl.steps}" )
                                    messages.back( argl.steps )
                                    raise CONTINUE()

                                elif "status" == argl.command:
                                    print( f"Status: {len(messages)}" )
                                    raise CONTINUE()


                                elif "save" == argl.command:
                                    fname = argl.filename
                                    with open( fname, "w" ) as fout:
                                        json.dump( messages, fout )
                                    raise CONTINUE()

                                elif "load" == argl.command:
                                    fname = argl.filename
                                    with open( fname, "r" ) as fin:
                                        messages = json.read( fin )
                                    raise CONTINUE()

                                elif "include" == argl.command:
                                    fname = argl.filename
                                    with open( fname, "r" ) as fin:
                                        you = fin.read()

                            except SystemExit:
                                print( "Sys Ex" )
                                pass

                    you = you.strip()

                    if Verbose:
                        print( C(Fore.LIGHTBLUE_EX) + you )

                    if len( you ) > 0:

                        if you[-1] not in (".", "!", "?"):
                            you += "."

                        messages.append( "user", you )

            except CONTINUE:
                if Verbose:
                    print( "CONTINUE" )

    except EXIT:
        if not Batch:
            print( C(Fore.LIGHTRED_EX) + "See U!" )
        exit()

    except:
        with open( Name + ".dump.ai", "w" ) as fout:
            fout.write( str( messages ) )
        raise
