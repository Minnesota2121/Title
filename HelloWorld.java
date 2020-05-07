import com.a51work6.Date;
import com.a51work6.ProtectedClass;

public class HelloWorld {
	
	public static void main(String[] args) {
		
		Date date = new Date();
		System.out.println(date);
		
		java.util.Date now = new java.util.Date();  //不能引入相同的类名，因此第二个Date需要在前面制定包名
		System.out.println(now);                    //区分now是java.util包中的Date，不是import引入的a51work6包中的Date
		
		
		ProtectedClass p = new ProtectedClass();
		p.printX();
		
		System.out.println("Hello World");
		
	}
}
